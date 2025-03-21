if __name__ == "__main__":
    # Test DataLoader
    dataloader = get_dataloader_g1(split="val", use_mask=True)
    for batch in dataloader:
        print(batch["input_edge"].shape, batch["gt_edge"].shape, batch["mask"].shape)
        break