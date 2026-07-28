PUT YOUR PHOTOS HERE — one subfolder per person.

Example:
  work/my_people/
      Sharath/       photo1.jpg  photo2.jpg  photo3.jpg
      Alex/          a.jpg  b.jpg  c.jpg
      Priya/         1.png  2.png  3.png

Use 3-5 clear, well-lit, front-facing photos per person (one face each).
Then enrol everyone:   python -m tools.my_faces register --dir work/my_people
Identify a new photo:  python -m tools.my_faces recognize --image somephoto.jpg
