def experiment_5_compare(df_fs, df_hybrid):
    def dispersion(df, h, f):
        return df.groupby("language")[[h, f]].std().mean().mean()

    d_fs = dispersion(df_fs, "H", "F")
    d_hy = dispersion(df_hybrid, "Hh", "Fh")

    print(f"Intra-language dispersion:")
    print(f"Fisher–Shannon: {d_fs:.4f}")
    print(f"Hybrid: {d_hy:.4f}")

    return {"FS_dispersion": d_fs, "Hybrid_dispersion": d_hy}
