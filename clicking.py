import pyautogui
import time
import random


next_button = {
    'x': 1002,
    'y': 1820
}
search_button = {
    'x': 83,
    'y': 978
}
playlist_location = {
    'x': 445,
    'y': 1153
}
first_song_location = {
    'x': 301,
    'y': 1397
}

playlist_name = "hacker music playlist"

low_time = 31
high_time = 63


def get_mouse_position():
    while True:
        x, y = pyautogui.position()
        print(x, y)
        time.sleep(1)


def play_next_song():
    while True:
        sleep_time = random.randint(low_time, high_time)
        while sleep_time > 0:
            print(sleep_time)
            time.sleep(1)
            sleep_time -= 1
        
        pyautogui.click(next_button['x'], next_button['y'])


def search_and_play(name):
    pyautogui.click(search_button['x'], search_button['y'])
    time.sleep(1)
    pyautogui.write(name)
    time.sleep(1)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.click(playlist_location['x'], playlist_location['y'])
    time.sleep(1)
    pyautogui.click(first_song_location['x'], first_song_location['y'])
    time.sleep(1)


#get_mouse_position()

search_and_play(playlist_name)
play_next_song()
