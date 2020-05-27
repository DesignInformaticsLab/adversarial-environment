def test():
    # load data

    vae.eval()
    load_param(vae)

    print('Evaluation')

    running_loss, mse_loss, kld_loss, no_of_batches = 0, 0, 0, 0
    i = 0
    file = open("dynamics/imgs_v2/Losses.txt", "w")
    file.write("Seed {}".format(args.seed))
    for index in BatchSampler(SubsetRandomSampler(range(len(images))), batch_size, False):
        with torch.no_grad():
            recon_images, _, mu, logvar = vae(images[index])

            loss, mse, kld = loss_fn(recon_images, images[index], mu, logvar)

        running_loss += loss.item()
        mse_loss += mse.item()
        kld_loss += kld.item()
        no_of_batches += 1

        print('loss: {} mse: {} kld: {}'.format(running_loss / no_of_batches, mse_loss / no_of_batches,
                                                kld_loss / no_of_batches))
        file.write("\nLoss for Batch {}: {} mse: {} kld: {}".format(i, loss.item(), mse.item(), kld.item()))

        num = np.random.randint(0, batch_size)
        bounds = np.random.randint(0, episodes - batch_size)
        with torch.no_grad():
            plt.title('Reconstruction')
            recon, _, _ = vae(images[bounds:bounds + batch_size])
            plt.imshow(recon[num].reshape((96, 96)), cmap='gray')
            plt.savefig('dynamics/imgs_v2/{}_Recon.png'.format(i))
        plt.title('Ground Truth')
        plt.imshow(images[bounds:bounds + batch_size][num].reshape((96, 96)), cmap='gray')
        plt.savefig('dynamics/imgs_v2/{}_GT.png'.format(i))

        i += 1

    print('Total Loss: {} Total mse: {} Total kld: {}'.format(running_loss, mse_loss, kld_loss))
    file.write("\nTotal Loss: {}".format(running_loss.mse_loss, kld_loss))
    file.close()
