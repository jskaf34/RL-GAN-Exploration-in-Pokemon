from pyboy import PyBoy
import numpy as np
import keyboard
from PIL import Image

# Initialize PyBoy
pyboy = PyBoy('jeu/PokemonRed.gb')
pyboy.set_emulation_speed(2)
with open("jeu/init_state_pokeball.state", "rb") as state_file:
    pyboy.load_state(state_file)

while not pyboy.tick():
    # pass
    if keyboard.is_pressed('t'):
    #     # Save the game state when 'S' key is pressed
    #     # with open("PokemonRed.state", "wb") as state_file:
    #     #     pyboy.save_state(state_file)
        
        # Test Preprocessing
        raw_observation = np.array(pyboy.screen_image())
        gray_obs = np.dot(raw_observation[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        image = Image.fromarray(raw_observation.astype(np.uint8))
        gray = Image.fromarray(gray_obs)
        image.save("original_image.png")
        gray.save("grayscaled_image.png")
        resized = gray.resize((140, 140))
        resized.save("resized_grayscale_image.png")
        res =image.resize((140, 140))
        res.save("resized.png")


# Close the emulator
pyboy.stop()