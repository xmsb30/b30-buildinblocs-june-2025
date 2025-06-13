# b30-buildinblocs-june-2025
```
!! LINK TO SLIDES: https://docs.google.com/presentation/d/1s0qnTVRW5fb5rurYUZSyWt1Kf7ApiaHIElQswJty0ls/edit?usp=sharing !!

Only use Python 3.10 or preferably 3.11 on Visual Studio Code.
You need to install the following libraries:
    tensorflow-cpu==2.15.0, opencv-python, pyautogui, numpy, matplotlib, Pillow

In your Visual Studio Code folder, you should have these three items:
    main.py,  
    haarcascade_frontalface_default.xml,
    eyes.tflite

Install the Google Chrome extension, Read Aloud: A Text TO Speech Voice Reader 2.20.0
(https://chromewebstore.google.com/detail/read-aloud-a-text-to-spee/hdhinadidafjejdhmfkjgnolgimiaplp?hl=en)

Enter chrome://extensions/shortcuts in your browser.
Find "Read Aloud: A Text to Speech Voice Reader".
Under "Activate the extension", set the shortcut to Alt + 0

Run main.py (ensure that camera access is enabled)
Within 10 seconds, switch to a news article site.

If squinting for more than 5 seconds, article will zoom in by 10%.
After 30 seconds of not squinting, article will zoom out by 10%.
(min zoom: 100%, max zoom: 190%)

Once the article has zoomed in 4 times, narrator mode will automatically activate.
To exit the program, press 'q' on the keyboard.
