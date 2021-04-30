import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

def visualizeGenerator(gen):
    img, mask = next(gen)
    fig = plt.figure(figsize=(10, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
    
    for i in range(2):
        cols = 2
        rows = math.ceil(img.shape[0]/cols)
        innerGrid = gridspec.GridSpecFromSubplotSpec(rows, cols,
                        subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(img.shape[0]):
            ax = plt.Subplot(fig, innerGrid[j])
            if(i==1):
                ax.imshow(img[j][:,:,:3], interpolation='none', origin='upper')
            else:
                ax.imshow(mask[j][:,:,0], interpolation='none', origin='upper')
                
            ax.axis('off')
            fig.add_subplot(ax)        
    plt.show()