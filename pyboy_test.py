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
        raw_observation = pyboy.screen_image()
        raw_observation.save("raw_observation.png")
        resized = raw_observation.resize((100, 100)).convert('L')
        resized.save("resized.png")
        gray = raw_observation.convert('L')
        gray.save("grayscale.png")
        gray_resized = gray.resize((100, 100))
        gray_resized.save("resized_grayscale.png")

# Close the emulator
pyboy.stop()