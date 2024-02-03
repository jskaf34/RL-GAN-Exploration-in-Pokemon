from pyboy import PyBoy
import numpy as np
import keyboard
from PIL import Image

# Initialize PyBoy
pyboy = PyBoy('jeu/PokemonRed.gb')
pyboy.set_emulation_speed(2)
with open("jeu/init_state_pokeball.state", "rb") as state_file:
    pyboy.load_state(state_file)

bool_ = True

while bool_ == True:
    # pass
    if keyboard.is_pressed('t'):
        while not keyboard.is_pressed('s') :
            if keyboard.is_pressed('p') :
                pyboy.tick()

    pyboy.tick()

    if keyboard.is_pressed('n') :
        bool_ = False

# Close the emulator
pyboy.stop()