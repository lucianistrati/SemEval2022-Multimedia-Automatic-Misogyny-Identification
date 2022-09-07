from x_transformers import Decoder, Encoder


def main():
    model = XTransformer(
        dim=512,
        enc_num_tokens=256,
        enc_depth=6,
        enc_heads=8,
        enc_max_seq_len=1024,
        dec_num_tokens=256,
        dec_depth=6,
        dec_heads=8,
        dec_max_seq_len=1024,
        tie_token_emb=True  # tie embeddings of encoder and decoder
    )

    src = torch.randint(0, 256, (1, 1024))
    src_mask = torch.ones_like(src).bool()
    tgt = torch.randint(0, 256, (1, 1024))
    tgt_mask = torch.ones_like(tgt).bool()

    loss = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)  # (1, 1024, 512)
    loss.backward()

    enc = Encoder(
        dim=512,
        depth=6,
        heads=8,
        attn_num_mem_kv=16  # 16 memory key / values
    )


if __name__ == "__main__":
    main()

