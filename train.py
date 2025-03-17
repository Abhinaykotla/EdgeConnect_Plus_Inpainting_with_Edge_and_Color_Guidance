if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from dataloader import get_dataloader_g1
    from g1_model import adversarial_loss, l1_loss
    from g1_model import EdgeGenerator, EdgeDiscriminator

    # Define loss functions
    def adversarial_loss(pred, target):
        """
        Computes adversarial loss using Binary Cross-Entropy with logits.
        """
        return F.binary_cross_entropy_with_logits(pred, target)

    def l1_loss(pred, target):
        """
        Computes L1 (pixel-wise) loss.
        """
        return F.l1_loss(pred, target)

    # Initialize models
    g1 = EdgeGenerator().cuda()
    d1 = EdgeDiscriminator().cuda()

    # Optimizers
    optimizer_g = torch.optim.Adam(g1.parameters(), lr=1e-4, betas=(0, 0.9))
    optimizer_d = torch.optim.Adam(d1.parameters(), lr=1e-4, betas=(0, 0.9))

    # Load dataset
    dataloader = get_dataloader_g1(split="train", use_mask=True)

    # Training Loop
    num_epochs = 50

    for epoch in range(num_epochs):
        total_g_loss = 0.0
        total_d_loss = 0.0

        for batch in dataloader:
            input_edges, gt_edges, mask = (
                batch["input_edge"].cuda(),   # Input edge (masked edges)
                batch["gt_edge"].cuda(), # Full-resolution ground truth edges
                batch["mask"].cuda()
            )

            ###### 🔹 Train Generator (G1) ######
            g1.train()
            pred_edge = g1(input_edges, mask)  # Generate edges

            # Compute L1 Loss with full-resolution edges
            g1_loss_l1 = l1_loss(pred_edge, gt_edges)

            # Compute Adversarial Loss using D1
            fake_pred = d1(input_edges, pred_edge)  # Fake pair: [input_edges, pred_edge]

            # Ensure target labels match D1's output shape ([B, 1, 30, 30])
            target_real = torch.ones_like(fake_pred)
            g1_loss_adv = adversarial_loss(fake_pred, target_real)

            loss_g = g1_loss_l1 + g1_loss_adv  # Combine L1 + Adversarial loss

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            ###### 🔹 Train Discriminator (D1) ######
            real_pred = d1(input_edges, gt_edges)  # Real pair: [input_edges, gt_edges]
            fake_pred_detached = d1(input_edges, pred_edge.detach())  # Fake pair (detached)

            target_fake = torch.zeros_like(fake_pred_detached)

            real_loss = adversarial_loss(real_pred, target_real)
            fake_loss = adversarial_loss(fake_pred_detached, target_fake)
            loss_d = (real_loss + fake_loss) / 2  # Average the loss

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # Track Losses
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()

        # Print epoch stats
        print(f"Epoch [{epoch+1}/{num_epochs}] - G1 Loss: {total_g_loss:.4f}, D1 Loss: {total_d_loss:.4f}")

