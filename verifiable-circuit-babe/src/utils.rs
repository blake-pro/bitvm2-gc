use ark_bn254::{Fq, Fr, G1Affine, G2Affine};
use ark_ec::{AffineRepr, CurveGroup};
use ark_groth16::VerifyingKey as Groth16VerifyingKey;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ripemd::{Digest as RipemdDigest, Ripemd160};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

pub fn groth16_vk_x(
    vk: &Groth16VerifyingKey<ark_bn254::Bn254>,
    public_inputs: &[Fr],
) -> Option<ark_bn254::G1Projective> {
    if vk.gamma_abc_g1.len() != public_inputs.len() + 1 {
        return None;
    }
    let mut acc = vk.gamma_abc_g1[0].into_group();
    for (i, x) in public_inputs.iter().enumerate() {
        acc += vk.gamma_abc_g1[i + 1].into_group() * *x;
    }
    Some(acc)
}

/// Position of the dynamic public input `x_d` in the full Groth16 public-input vector.
///
/// Ziren's wrap circuit exposes `[vkey_hash, committed_values_digest, vk_root]`, so `x_d` sits at
/// index 1 with `vkey_hash` before it and `vk_root` after it. The same holds for the 2-input
/// `DummyMulCircuit` used in tests (`[static, dynamic]`).
pub const DYNAMIC_INPUT_IDX: usize = 1;

/// Index into `gamma_abc_g1` of the base point `L_D` of the dynamic input (`gamma_abc_g1[0]` is the
/// constant term).
pub const DYNAMIC_BASE_IDX: usize = DYNAMIC_INPUT_IDX + 1;

/// `P_S = L_0 + Σ x_S[i]·L_{pos(i)}`: the static part of `vk_x`.
///
/// `static_inputs` lists every public input except the dynamic one, in circuit order. Returns
/// `None` unless the vk has exactly one base point per static input plus `L_0` and `L_D`.
pub fn static_vk_x(
    vk: &Groth16VerifyingKey<ark_bn254::Bn254>,
    static_inputs: &[Fr],
) -> Option<ark_bn254::G1Projective> {
    if static_inputs.len() < DYNAMIC_INPUT_IDX || vk.gamma_abc_g1.len() != static_inputs.len() + 2 {
        return None;
    }
    let mut acc = vk.gamma_abc_g1[0].into_group();
    for (i, x) in static_inputs.iter().enumerate() {
        // Position of this static input once the dynamic one is slotted in.
        let pos = if i < DYNAMIC_INPUT_IDX { i } else { i + 1 };
        acc += vk.gamma_abc_g1[pos + 1].into_group() * *x;
    }
    Some(acc)
}

/// Full ordered public-input vector: `x_d` is inserted at [`DYNAMIC_INPUT_IDX`] among the statics.
pub fn assemble_public_inputs(static_inputs: &[Fr], x_d: Fr) -> Vec<Fr> {
    let (before, after) = static_inputs.split_at(DYNAMIC_INPUT_IDX.min(static_inputs.len()));
    before.iter().copied().chain(std::iter::once(x_d)).chain(after.iter().copied()).collect()
}

pub fn g2_project_to_ser(p: ark_bn254::G2Projective) -> Vec<u8> {
    let mut out = Vec::new();
    p.into_affine().serialize_compressed(&mut out).expect("serialize g2");
    out
}

pub fn g1_project_to_ser(p: ark_bn254::G1Projective) -> Vec<u8> {
    let mut out = Vec::new();
    p.into_affine().serialize_compressed(&mut out).expect("serialize g1");
    out
}

/// Non-identity, on-curve, correct-subgroup check for an already-parsed G1 point.
pub fn g1affine_is_valid(a: &G1Affine) -> bool {
    !a.is_zero() && a.is_on_curve() && a.is_in_correct_subgroup_assuming_on_curve()
}

pub fn g1_from_ser_checked(v: &[u8]) -> Option<ark_bn254::G1Projective> {
    let a = G1Affine::deserialize_compressed(v).ok()?;
    if !g1affine_is_valid(&a) {
        return None;
    }
    Some(a.into_group())
}

pub fn g2_from_ser_checked(v: &[u8]) -> Option<ark_bn254::G2Projective> {
    let a = G2Affine::deserialize_compressed(v).ok()?;
    if a.is_zero() || !a.is_on_curve() || !a.is_in_correct_subgroup_assuming_on_curve() {
        return None;
    }
    Some(a.into_group())
}

/// Encode pi1 and x_d into a 96-byte Wots96 message.
/// Layout: pi1.x LE-32 || pi1.y LE-32 || x_d LE-32.
/// Bit packing: bits[i] = (msg[i/8] >> (i%8)) & 1  (LSB-first within each byte).
/// Bits 254-255 of each 32-byte chunk are always 0 (BN254 fields are < 2^254).
pub fn pi1_xd_to_wots96_msg(pi1: &G1Affine, x_d: Fr) -> [u8; 96] {
    let mut msg = [0u8; 96];
    let mut tmp = Vec::new();

    pi1.x.serialize_uncompressed(&mut tmp).expect("serialize pi1.x");
    msg[..32].copy_from_slice(&tmp);

    tmp.clear();
    pi1.y.serialize_uncompressed(&mut tmp).expect("serialize pi1.y");
    msg[32..64].copy_from_slice(&tmp);

    tmp.clear();
    x_d.serialize_uncompressed(&mut tmp).expect("serialize x_d");
    msg[64..96].copy_from_slice(&tmp);

    msg
}

pub fn ro_from_pairing_bytes(seed: &[u8], msg_len: usize) -> Vec<u8> {
    let key = h_256(seed);
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&h_256(&[b"babe-we-known-pi1-ro-nonce".as_slice(), seed].concat())[..12]);
    derive_stream_xor_keyed(key, nonce, msg_len)
}

fn derive_stream_xor_keyed(key: [u8; 32], nonce: [u8; 12], msg_len: usize) -> Vec<u8> {
    let mut out = vec![0u8; msg_len];
    let mut ctr: u32 = 0;
    let mut off = 0usize;
    while off < msg_len {
        let mut blk = b"babe-we-stream".to_vec();
        blk.extend_from_slice(&key);
        blk.extend_from_slice(&nonce);
        blk.extend_from_slice(&ctr.to_le_bytes());
        let hblk = h_256(&blk);
        let take = core::cmp::min(32, msg_len - off);
        out[off..off + take].copy_from_slice(&hblk[..take]);
        ctr += 1;
        off += take;
    }
    out
}

#[inline(always)]
/// SHA256
pub fn h_256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// RIPEMD160
pub fn h_160(data: &[u8]) -> [u8; 20] {
    let mut hasher = Ripemd160::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// h_msg = RIPEMD160(SHA256(data)) — the hashlock value.
pub fn derive_hashlock(secret: &[u8]) -> [u8; 20] {
    h_160(&h_256(&secret))
}

/// `serde` shim for `Fq` via its compressed `ark-serialize` byte form.
/// Use with `#[serde(serialize_with = "crate::utils::serialize_fq", deserialize_with = "crate::utils::deserialize_fq")]`.
pub fn serialize_fq<S: Serializer>(val: &Fq, serializer: S) -> Result<S::Ok, S::Error> {
    let mut bytes = Vec::new();
    val.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
    bytes.serialize(serializer)
}

pub fn deserialize_fq<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Fq, D::Error> {
    let bytes = Vec::<u8>::deserialize(deserializer)?;
    Fq::deserialize_compressed(bytes.as_slice()).map_err(serde::de::Error::custom)
}

pub fn serialize_g1affine<S: Serializer>(val: &G1Affine, serializer: S) -> Result<S::Ok, S::Error> {
    let mut bytes = Vec::new();
    val.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
    bytes.serialize(serializer)
}

pub fn deserialize_g1affine<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<G1Affine, D::Error> {
    let bytes = Vec::<u8>::deserialize(deserializer)?;
    G1Affine::deserialize_compressed(bytes.as_slice()).map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn test_groth16_vk_x() {
        let mut rng = ChaCha12Rng::seed_from_u64(1);
        let g0 = ark_bn254::G1Projective::rand(&mut rng).into_affine();
        let g1 = ark_bn254::G1Projective::rand(&mut rng).into_affine();
        let g2 = ark_bn254::G1Projective::rand(&mut rng).into_affine();
        let vk = Groth16VerifyingKey::<ark_bn254::Bn254> {
            gamma_abc_g1: vec![g0, g1, g2],
            ..Default::default()
        };

        let x1 = Fr::from(3u64);
        let x2 = Fr::from(5u64);
        let expected =
            (g0.into_group() + g1.into_group() * x1 + g2.into_group() * x2).into_affine();
        let actual = groth16_vk_x(&vk, &[x1, x2]).unwrap().into_affine();
        assert_eq!(actual, expected);

        // length mismatch: public_inputs.len() + 1 != gamma_abc_g1.len()
        assert!(groth16_vk_x(&vk, &[x1]).is_none());
        assert!(groth16_vk_x(&vk, &[x1, x2, Fr::from(7u64)]).is_none());
    }

    /// Ziren v2.0 layout: `[vkey_hash, committed_values_digest, vk_root]`, i.e. a 4-point vk with
    /// the dynamic input in the middle.
    #[test]
    fn test_static_vk_x_with_three_public_inputs() {
        let mut rng = ChaCha12Rng::seed_from_u64(3);
        let l: Vec<G1Affine> =
            (0..4).map(|_| ark_bn254::G1Projective::rand(&mut rng).into_affine()).collect();
        let vk = Groth16VerifyingKey::<ark_bn254::Bn254> {
            gamma_abc_g1: l.clone(),
            ..Default::default()
        };

        let vkey_hash = Fr::from(11u64);
        let vk_root = Fr::from(13u64);
        let x_d = Fr::from(17u64);
        let statics = [vkey_hash, vk_root];

        // P_S skips L_2, which belongs to the dynamic input.
        let expected_ps =
            l[0].into_group() + l[1].into_group() * vkey_hash + l[3].into_group() * vk_root;
        let p_s = static_vk_x(&vk, &statics).unwrap();
        assert_eq!(p_s.into_affine(), expected_ps.into_affine());

        // P_S + x_d·L_D equals the generic vk_x over the full, correctly ordered input vector.
        let full = assemble_public_inputs(&statics, x_d);
        assert_eq!(full, vec![vkey_hash, x_d, vk_root]);
        let vk_x = groth16_vk_x(&vk, &full).unwrap();
        assert_eq!(
            (p_s + vk.gamma_abc_g1[DYNAMIC_BASE_IDX] * x_d).into_affine(),
            vk_x.into_affine()
        );

        // A vk with the old 3-point layout does not match two static inputs, and vice versa.
        let old_vk = Groth16VerifyingKey::<ark_bn254::Bn254> {
            gamma_abc_g1: l[..3].to_vec(),
            ..Default::default()
        };
        assert!(static_vk_x(&old_vk, &statics).is_none());
        assert!(static_vk_x(&vk, &statics[..1]).is_none());
        assert!(static_vk_x(&vk, &[]).is_none());
    }

    /// The 2-input layout used by `DummyMulCircuit` (`[static, dynamic]`) keeps working.
    #[test]
    fn test_static_vk_x_with_two_public_inputs() {
        let mut rng = ChaCha12Rng::seed_from_u64(4);
        let l: Vec<G1Affine> =
            (0..3).map(|_| ark_bn254::G1Projective::rand(&mut rng).into_affine()).collect();
        let vk = Groth16VerifyingKey::<ark_bn254::Bn254> {
            gamma_abc_g1: l.clone(),
            ..Default::default()
        };

        let x_s = Fr::from(5u64);
        let x_d = Fr::from(7u64);
        let p_s = static_vk_x(&vk, &[x_s]).unwrap();
        assert_eq!(p_s.into_affine(), (l[0].into_group() + l[1].into_group() * x_s).into_affine());
        assert_eq!(assemble_public_inputs(&[x_s], x_d), vec![x_s, x_d]);
    }

    #[test]
    fn test_g1_from_ser_checked() {
        let mut rng = ChaCha12Rng::seed_from_u64(2);
        let p = ark_bn254::G1Projective::rand(&mut rng).into_affine();
        let mut valid_bytes = Vec::new();
        p.serialize_compressed(&mut valid_bytes).unwrap();
        assert_eq!(g1_from_ser_checked(&valid_bytes).unwrap().into_affine(), p);

        let mut identity_bytes = Vec::new();
        G1Affine::zero().serialize_compressed(&mut identity_bytes).unwrap();
        assert!(g1_from_ser_checked(&identity_bytes).is_none());

        assert!(g1_from_ser_checked(&[0xFFu8; 32]).is_none());
    }

    #[test]
    fn test_g2_from_ser_checked() {
        let mut rng = ChaCha12Rng::seed_from_u64(3);
        let p = ark_bn254::G2Projective::rand(&mut rng).into_affine();
        let mut valid_bytes = Vec::new();
        p.serialize_compressed(&mut valid_bytes).unwrap();
        assert_eq!(g2_from_ser_checked(&valid_bytes).unwrap().into_affine(), p);

        let mut identity_bytes = Vec::new();
        G2Affine::zero().serialize_compressed(&mut identity_bytes).unwrap();
        assert!(g2_from_ser_checked(&identity_bytes).is_none());

        assert!(g2_from_ser_checked(&[0xFFu8; 64]).is_none());
    }

    #[test]
    fn test_pi1_xd_to_wots96_msg_byte_layout() {
        let mut rng = ChaCha12Rng::seed_from_u64(4);
        let pi1 = ark_bn254::G1Projective::rand(&mut rng).into_affine();
        let x_d = Fr::rand(&mut rng);
        let msg = pi1_xd_to_wots96_msg(&pi1, x_d);

        let mut x_bytes = Vec::new();
        pi1.x.serialize_uncompressed(&mut x_bytes).unwrap();
        let mut y_bytes = Vec::new();
        pi1.y.serialize_uncompressed(&mut y_bytes).unwrap();
        let mut xd_bytes = Vec::new();
        x_d.serialize_uncompressed(&mut xd_bytes).unwrap();

        assert_eq!(&msg[0..32], x_bytes.as_slice());
        assert_eq!(&msg[32..64], y_bytes.as_slice());
        assert_eq!(&msg[64..96], xd_bytes.as_slice());
    }

    #[test]
    fn test_ro_from_pairing_bytes() {
        let seed1 = b"seed-one".as_slice();
        let seed2 = b"seed-two".as_slice();

        let out1a = ro_from_pairing_bytes(seed1, 40);
        let out1b = ro_from_pairing_bytes(seed1, 40);
        let out2 = ro_from_pairing_bytes(seed2, 40);

        assert_eq!(out1a.len(), 40);
        assert_eq!(out1a, out1b, "same seed must be deterministic");
        assert_ne!(out1a, out2, "different seeds must diverge");

        // Stream is CTR-based over blocks derived from (key, nonce) alone, so a shorter
        // request must be a prefix of a longer one for the same seed.
        let out_short = ro_from_pairing_bytes(seed1, 5);
        assert_eq!(out_short.as_slice(), &out1a[..5]);
    }

    #[test]
    fn test_serde_shims_roundtrip() {
        #[derive(Serialize, Deserialize)]
        struct FqWrap(
            #[serde(serialize_with = "serialize_fq", deserialize_with = "deserialize_fq")] Fq,
        );
        #[derive(Serialize, Deserialize)]
        struct G1Wrap(
            #[serde(
                serialize_with = "serialize_g1affine",
                deserialize_with = "deserialize_g1affine"
            )]
            G1Affine,
        );

        let mut rng = ChaCha12Rng::seed_from_u64(5);

        let fq = Fq::rand(&mut rng);
        let bytes = bincode::serialize(&FqWrap(fq)).unwrap();
        let back: FqWrap = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.0, fq);

        let g1 = ark_bn254::G1Projective::rand(&mut rng).into_affine();
        let bytes = bincode::serialize(&G1Wrap(g1)).unwrap();
        let back: G1Wrap = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.0, g1);
    }
}
