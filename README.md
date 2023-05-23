# Vietnamese Detection
Vietnamese is an implementation of facial recognition, detection of facial attributes (age, gender, emotion and vietnamese detection) for python.
The repository provides a script to run Face Info with the webcam or by entering the path of an image.
This implementation allows recognition of multiple faces and the registration of new users for facial recognition.

# How to install:
## Download model and save it in folder weight
- **Emotion detection:** https://drive.google.com/file/d/1SdMDBS7zz8OA91PmtQTmXDVFZAZJmg2L/view?usp=sharing
- **Age detection:** https://drive.google.com/file/d/1awkrhXlqGackc-2X8tsUy9_mYxzHL0Ha/view?usp=sharing
- **Gender detection:** https://drive.google.com/file/d/1oVCDdVhFdaSXKUDC0Of0OcGOIOMHkbYR/view?usp=sharing
- **Vietnamese detection:** https://drive.google.com/file/d/1ZNvVlA3ef5ogw2dNU4o0V13xMrPS7MFn/view?usp=sharing
<pre><code>pip install -r requirements.txt </code></pre>


# How to run:
The code is tested in python 3.7.8 and macOS Catalina
<pre><code>python Face_info.py --input webcam </code></pre>



running over an image
<pre><code>python Face_info.py --input image --path_im data_test/friends.jpg </code></pre>


# Add new faces to the database (facial recognition)
You can add new users to the faces database simply by adding the person's photo in the **images_db** folder, for the registry to work correctly, only the person of interest should appear in the photo.

