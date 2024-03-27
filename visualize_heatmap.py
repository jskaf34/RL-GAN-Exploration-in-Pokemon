import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image



def make_all_coords_arrays(df):
    return np.array(df[['x', 'y', 'm']].to_numpy().astype(np.uint8))

def game_coord_to_pixel_coord(
    x, y, map_idx, base_y):
    
    global_offset = np.array([1056-16*12, 331]) #np.array([790, -29])
    map_offsets = {
        0: np.array([0,211]), # pallet town
        1: np.array([-10, 138]), # viridian
        2: np.array([-10, 180]), # pewter
        12: np.array([0, 175]), # route 1
        13: np.array([0, 64]), # route 2
        14: np.array([30, 172]), # Route 3
        15: np.array([80, 190]), #Route 4
        33: np.array([-50, 145]), # route 22
        37: np.array([-9, 216]), # red house first
        38: np.array([-9, 25-32]), # red house second
        39: np.array([9+12, 207]), # blues house
        40: np.array([25-4, 217]), # oaks lab
        41: np.array([30, 162]), # Pokémon Center (Viridian City)
        42: np.array([30, 155]), # Poké Mart (Viridian City)
        43: np.array([30, 137]), # School (Viridian City)
        44: np.array([30, 147]), # House 1 (Viridian City)
        47: np.array([21,136]), # Gate (Viridian City/Pewter City) (Route 2)
        49: np.array([21,108]), # Gate (Route 2)
        50: np.array([21,102]), # Gate (Route 2/Viridian Forest) (Route 2)
        51: np.array([-35, 73]), # viridian forest
        52: np.array([-10, 189]), # Pewter Museum (floor 1)
        53: np.array([-10, 198]), # Pewter Museum (floor 2)
        54: np.array([-21, 169]), #Pokémon Gym (Pewter City)
        55: np.array([-19, 177]), #House with disobedient Nidoran♂ (Pewter City)
        56: np.array([-30, 163]), #Poké Mart (Pewter City)
        57: np.array([-19, 177]), #House with two Trainers (Pewter City)
        58: np.array([-25, 154]), # Pokémon Center (Pewter City)
        59: np.array([83, 227]), # Mt. Moon (Route 3 entrance)
        60: np.array([123, 227]), # Mt. Moon
        61: np.array([152, 227]), # Mt. Moon
        68: np.array([65, 190]), # Pokémon Center (Route 4)
        193: None # Badges check gate (Route 22)
    }
    if map_idx in map_offsets.keys():
        offset = map_offsets[map_idx]
    else:
        offset = np.array([0,0])
        x, y = 0, 0
    coord = global_offset + 16*(offset + np.array([x,y]))
    coord[1] = base_y - coord[1]
    return coord

def blend_overlay(background, over):
    al = over[...,3].reshape(over.shape[0], over.shape[1], 1)
    ba = (255-al)/255
    oa = al/255
    return (background[..., :3]*ba + over[..., :3]*oa).astype(np.uint8)

def main(args): 
    datapath = args.datapath

    df_data = pd.read_csv(datapath)

    base_coords = make_all_coords_arrays(df_data)

    main_map = np.array(Image.open('pokemap_full.png'))
    start_bg = main_map.copy()
    overlay = np.zeros_like(start_bg, dtype=np.uint8)

    pixel_coords = np.zeros((base_coords.shape[0], 2))

    for i, row in enumerate(base_coords):
        x, y, m = row
        x_img, y_img = game_coord_to_pixel_coord(x, y, m, overlay.shape[1])
        pixel_coords[i,:] = np.array([x_img, y_img]).astype(int)

    img_width = 4000
    img_height = 4000

    plt.figure(figsize=(10, 10))  
    plt.imshow(blend_overlay(start_bg, overlay), cmap='gray', extent=[0, img_width, 0, img_height])

    # Plot the density of points as a heatmap
    plt.hexbin(pixel_coords[:, 0], 
        pixel_coords[:, 1], 
        gridsize=250, 
        cmap='plasma', 
        alpha=0.8, 
        mincnt=1, 
        extent=[0, img_width, 0, img_height]
    ) 
    plt.colorbar(label='Density')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Density Heatmap Overlay on Image')

    plt.xlim(0, img_width)  
    plt.ylim(0, img_height) 
    
    if args.output_name is not None:
        plt.savefig(args.output_name, dpi=300, bbox_inches='tight') 
    if args.show:
        plt.show()

if __name__=="__main__": 
    parser = argparse.ArgumentParser(description="Plotting and/or saving heatmap from data.")
    
    parser.add_argument(
        '--datapath', 
        type=str, 
        help="The path of the data to be processed"
    )
    parser.add_argument(
        '--show', 
        action="store_true",
        help='Whether or not to show the heatmap'
    )
    parser.add_argument(
        '--output_name', 
        type=str,
        default=None,
        help='The path to save the heatmap'
    )

    args = parser.parse_args()

    main(args)