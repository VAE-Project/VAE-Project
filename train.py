from utils import to_device
import numpy as np
import torch


def train_autoencoder(model, train_loader, val_loader, optimizer, args):

    # Init
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    for epoch in range(args.epochs):

        running_loss = 0
        running_val_loss = 0

        for idx, batch in enumerate(train_loader):
            # Prediction
            target = to_device(batch, args.device)
            if model.p_zero:  # Denoising AE
                mask = to_device(torch.rand(size=batch.shape) < model.p_zero, args.device)
                noisy_target = target.masked_fill(mask, 0)
                reconstruction = model(noisy_target)
            else:
                reconstruction = model(target)

            # Loss
            optimizer.zero_grad()
            loss = model.criterion(reconstruction, target)
            running_loss += loss.item()

            # Backprop and params' update
            loss.backward()
            optimizer.step()

        for idx, batch in enumerate(val_loader):
            # Prediction
            target = to_device(batch, args.device)
            reconstruction = model(target)

            # Loss
            loss = model.criterion(reconstruction, target)
            running_val_loss += loss.item()

        # Average loss over the batches during the training
        model.logs["train loss"].append(running_loss/train_size)
        model.logs["val loss"].append(running_val_loss/val_size)



def train_gan(model, train_loader, optimizer_D, optimizer_G, args):
    # Initialization
    Tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor
    model.to(args.device)
    
    for epoch in range(args.epochs):

        for idx, batch in enumerate(train_loader):

            batch = to_device(batch, args.device)
            batch_size = batch.shape[0]
            # Train discriminator
            optimizer_D.zero_grad()
            # Noise
            z = Tensor(np.random.normal(
                0, 1, size=(batch_size, args.random_dim)), device=args.device)
            # Generate synthetic examples
            batch_synthetic = model.G(z).detach()  # No gradient for generator's parameters
            # Discriminator outputs
            y_real = model.D(batch)
            y_synthetic = model.D(batch_synthetic)
            # Gradient penalty
            gradient_penalty = model.D.compute_gradient_penalty(batch.data, batch_synthetic.data)
            # Loss & Update
            loss_D = model.D.loss(y_real, y_synthetic, gradient_penalty)
            loss_D.backward()
            optimizer_D.step()

            # Train generator ever n_critic iterations
            if idx % args.n_critic == 0:
                # The loss function at this point is an approximate of EM distance
                model.logs["approx. EM distance"].append(loss_D.item())
                optimizer_G.zero_grad()
                # Generate synthetic examples
                batch_synthetic = model.G(z)
                # Loss & Update
                loss_G = model.G.loss(model.D(batch_synthetic))
                loss_G.backward()
                optimizer_G.step()
                if args.verbose and idx % 100 == 0:
                    print(
                        f"Epoch {epoch}, Iteration {idx}, Appriximation of EM distance: {loss_D.item()}")



def train_vae(model, train_loader, val_loader, optimizer, args):
    model.to(args.device)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    for epoch in range(args.epochs):
        # training
        model.train()
        running_loss = 0
        for idx, batch in enumerate(train_loader):
            batch = to_device(batch, args.device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(batch)
            loss = model.loss(batch, reconstruction, mu, logvar)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

        # validation
        model.eval()
        running_val_loss = 0
        for idx, batch in enumerate(val_loader):
            batch = to_device(batch, args.device)
            reconstruction, mu, logvar = model(batch)
            loss = model.loss(batch, reconstruction, mu, logvar)
            running_val_loss += loss.item()

        model.logs["train loss"].append(running_loss/train_size)
        model.logs["val loss"].append(running_val_loss/val_size)    
