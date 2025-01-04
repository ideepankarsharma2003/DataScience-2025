Imagine you're playing with a toy car on a track. You want to keep an eye on your car as it moves around, even if it goes behind obstacles or other toys. The Deep SORT algorithm helps computers do something similar: it watches and follows multiple moving objects, like cars or people, in videos, even when they disappear and reappear.

**How Does Deep SORT Work?**

1. **Seeing the Objects:** First, the computer uses its "eyes" (cameras) to spot all the moving objects in a video frame. This is like you looking at your toy car and other toys on the track.

2. **Remembering Appearances:** Each object has unique features—like color, shape, or size—that make it distinct. The computer notes these features to recognize the same object in future frames, even if it moves or hides behind something. This step is crucial for distinguishing between different objects.

3. **Predicting Movement:** The computer uses a mathematical tool called a Kalman filter to guess where each object will move next. It's like predicting that your toy car will continue along the track in a certain direction and speed.

4. **Matching Predictions with Reality:** In the next video frame, the computer compares its predictions with what it actually sees. If an object appears where the computer expected, it confirms the object's identity. If not, it updates its prediction based on the new information.

5. **Handling Disappearances:** If an object goes out of sight (like your toy car going behind a couch), the computer doesn't immediately forget about it. It waits for a while, using its memory of the object's appearance and movement to recognize it when it comes back into view.

**Mathematical Intuition Behind Deep SORT**

- **Kalman Filter:** This is a mathematical formula that helps the computer make educated guesses about an object's future position based on its past movement. It considers possible errors in measurement and movement to improve its predictions.

- **Cosine Distance:** To recognize if an object in a new frame is the same as one seen before, the computer calculates the "cosine distance" between their appearance features. A smaller distance means the objects look more alike, helping the computer match identities accurately.

By combining these mathematical tools, Deep SORT effectively tracks multiple objects in videos, maintaining their identities even through occlusions or when they leave and re-enter the frame.

For a visual explanation, you might find this video helpful:


[![How DeepSORT works?](https://img.youtube.com/vi/LbyqsoLJu5Q/0.jpg)](https://www.youtube.com/watch?v=LbyqsoLJu5Q)

