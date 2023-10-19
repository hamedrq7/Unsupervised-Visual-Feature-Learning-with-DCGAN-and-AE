import torch 
import numpy as np
import os 
import matplotlib.pyplot as plt


def make_dir(dir_name: str):
    """
    creates directory "dir_name" if it doesn't exists
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def plot_recons_samples(recons, images, path_to_save: str, filename: str, title: str): 
    assert recons.shape[0] == 64, 'plotting 8x8 canvas, give 64 recons'
    assert images.shape[0] == 64, 'plotting 8x8 canvas, give 64 images'
    
    rat = 1 / 16
    rat_white = 0.04
    width_ratios = (np.ones((17)) * rat) - (rat_white / 16)
    width_ratios[8] = rat_white

    w = 33.28  # 16.64 x 32
    h = 16     # 8     x 32
    fig, axs = plt.subplots(nrows=8, ncols=17, figsize=(w, h), 
                            gridspec_kw={'width_ratios': width_ratios.tolist()},
                            )

    for row in range(8):
        for col in range(17): 
            if col < 8:
                i = row
                j = col
                axs[row, col].imshow(recons[i*8 + j])
            # elif col == 8:
            #     axs[row, col].imshow(X1[0 + 0])
            elif col > 8:
                i = row
                j = col - 9
                axs[row, col].imshow(images[i*8 + j])
            
            axs[row, col].axis('off') 
            # axs[row, col].set_xticklabels([])
            # axs[row, col].set_yticklabels([])
            # axs[row, col].set_aspect('equal')
    
    
    plt.subplots_adjust(wspace=0, hspace=0) # , left=0, right=1, bottom=0, top=1
    fig.suptitle(f'{title}', fontsize=32)
    # plt.tight_layout()
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # fig.subplots_adjust(hspace=0)
    # plt.show()
    make_dir(path_to_save)
    plt.savefig(f'{path_to_save}/{filename}.jpg') # , bbox_inches="tight"



def custom_plot_loss(loss_hist, phase_list, title: str, dir: str): 
    fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize=[7, 6], dpi=100)

    for phase in phase_list: 
        lowest_loss_x = np.argmin(np.array(loss_hist[phase]))
        lowest_loss_y = loss_hist[phase][lowest_loss_x]
        
        ax1.annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
        ax1.plot(loss_hist[phase], '-x', label=f'{phase} loss', markevery = [lowest_loss_x])

        ax1.set_xlabel(xlabel='epochs')
        ax1.set_ylabel(ylabel='loss')

        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax1.legend()
        ax1.label_outer()

    fig.suptitle(f'{title}')

    make_dir(dir)
    plt.savefig(f'{dir}/losses.jpg')
    plt.clf()
 

def custpm_plot_clf_acc(acc_hist, AE_epoch, phase_list, title: str, dir: str): 
    fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize=[7, 6], dpi=100)

    for phase in phase_list: 
        highest_acc_x = np.argmax(np.array(acc_hist[phase]))
        highest_acc_y = acc_hist[phase][highest_acc_x]
        
        ax1.annotate("{:.4f}".format(highest_acc_y), [highest_acc_x, highest_acc_y])
        ax1.plot(acc_hist[phase], '-x', label=f'{phase} loss', markevery = [highest_acc_x])

        ax1.set_xlabel(xlabel='epochs')
        ax1.set_ylabel(ylabel='loss')

        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax1.legend()
        ax1.label_outer()

    fig.suptitle(f'{title}')

    make_dir(dir)
    plt.savefig(f'{dir}/acc-AE_epoch_{AE_epoch}.jpg')
    plt.clf()


def custom_plot_training_stats(acc_hist, loss_hist, phase_list, title: str, dir: str): 
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=[14, 6], dpi=100)

    for phase in phase_list: 
        lowest_loss_x = np.argmin(np.array(loss_hist[phase]))
        lowest_loss_y = loss_hist[phase][lowest_loss_x]
        
        ax1.annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
        ax1.plot(loss_hist[phase], '-x', label=f'{phase} loss', markevery = [lowest_loss_x])

        ax1.set_xlabel(xlabel='epochs')
        ax1.set_ylabel(ylabel='loss')

        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax1.legend()
        ax1.label_outer()

    # acc: 
    for phase in phase_list:
        highest_acc_x = np.argmax(np.array(acc_hist[phase]))
        highest_acc_y = acc_hist[phase][highest_acc_x]
        
        ax2.annotate("{:.4f}".format(highest_acc_y), [highest_acc_x, highest_acc_y])
        ax2.plot(acc_hist[phase], '-x', label=f'{phase} loss', markevery = [highest_acc_x])

        ax2.set_xlabel(xlabel='epochs')
        ax2.set_ylabel(ylabel='acc')

        ax2.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax2.legend()
        #ax2.label_outer()

    fig.suptitle(f'{title}')

    make_dir(dir)
    plt.savefig(f'{dir}/AE acc-loss.jpg')
    plt.clf()

def cutsom_plot_GAN(Loss_G_hist, D_Gz_hist, Loss_D_hist, Dx_hist, title: str, dir: str, filename: str = 'GAN stats'):
    # Loss_G_hist -> decrease
    # Loss_D_hist -> decrease? 
    # Dx_hist     -> This should start close to 1 then theoretically converge to 0.5 when G gets better  - decrease
    # D_Gz_hist   -> These numbers should start near 0 and converge to 0.5 as G gets better - increase

    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize=[14, 12], dpi=100)

    # plot losses
    plt_indx = 0
    loss_hist = {'Loss G': Loss_G_hist, 'Loss D': Loss_D_hist}
    for phase in ['Loss G', 'Loss D']: 
        lowest_loss_x = np.argmin(np.array(loss_hist[phase]))
        lowest_loss_y = loss_hist[phase][lowest_loss_x]
        
        axs[0][plt_indx].annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
        axs[0][plt_indx].plot(loss_hist[phase], '-x', label=f'{phase}', markevery = [lowest_loss_x])

        axs[0][plt_indx].set_xlabel(xlabel='epochs')
        axs[0][plt_indx].set_ylabel(ylabel=f'{phase}')

        axs[0][plt_indx].grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        axs[0][plt_indx].legend()
        # axs[0][plt_indx].label_outer()
        plt_indx += 1

    # plot Dx
    plt_indx = 0
    lowest_Dx_x = np.argmin(np.array(Dx_hist))
    lowest_Dx_y = Dx_hist[lowest_Dx_x]
    
    axs[1][plt_indx].annotate("{:.4f}".format(lowest_Dx_y), [lowest_Dx_x, lowest_Dx_y])
    axs[1][plt_indx].plot(Dx_hist, '-x', label=f'Dx', markevery = [lowest_Dx_x])

    axs[1][plt_indx].set_xlabel(xlabel='epochs')
    axs[1][plt_indx].set_ylabel(ylabel='Dx')

    axs[1][plt_indx].grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
    axs[1][plt_indx].legend()
    # axs[1][plt_indx].label_outer()
    plt_indx += 1

    # plot D_Gz
    highest_D_Gz_x = np.argmax(np.array(D_Gz_hist))
    highest_D_Gz_y = D_Gz_hist[highest_D_Gz_x]
    
    axs[1][plt_indx].annotate("{:.4f}".format(highest_D_Gz_y), [highest_D_Gz_x, highest_D_Gz_y])
    axs[1][plt_indx].plot(D_Gz_hist, '-x', label=f'D Gz', markevery = [highest_D_Gz_x])

    axs[1][plt_indx].set_xlabel(xlabel='epochs')
    axs[1][plt_indx].set_ylabel(ylabel='D Gz')

    axs[1][plt_indx].grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
    axs[1][plt_indx].legend()
    # axs[1][plt_indx].label_outer()

    fig.suptitle(f'{title}')

    make_dir(dir)
    plt.savefig(f'{dir}/{filename}.jpg')
    plt.clf()
