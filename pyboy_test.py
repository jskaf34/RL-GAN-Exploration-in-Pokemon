from pyboy import PyBoy
import numpy as np
import keyboard

# Initialize PyBoy
pyboy = PyBoy('jeu/PokemonRed.gb')
pyboy.set_emulation_speed(2)
with open("PokemonRed4.state", "rb") as state_file:
    pyboy.load_state(state_file)

while not pyboy.tick():
    # pass
    if keyboard.is_pressed('t'):
        # Save the game state when 'S' key is pressed
        with open("PokemonRed.state", "wb") as state_file:
            pyboy.save_state(state_file)
        # raw_observation = np.array(pyboy.screen_image())
        # print(raw_observation.shape)


# Close the emulator
pyboy.stop()