# Instructions

1) Be sure to have demo_live.exe and yunet.onnx (the file of the neural network) in the same folder.
2) Open command line 
3) Go into the folder of demo_live.exe with cd command 
4) Use the following command, where rstp://[camera_ip] is the rtsp link :

        demo_live.exe -v rstp://[camera_ip]

5) To stop the execution of the algorithm, use ctrl+c

### IN CASE OF SLOW OUTPUT

The bigger the input images are, the slower is the algorithm and the less FPS you will get.
To face this problem, we can rescale the frames from the video using this command :

          demo_live.exe -v rstp://[camera_ip] -sc 0.5

So to have a faster output, the number after "-sc" should be between 0 and 1. 
The closer it is to 0, the faster it is but it will also be less accurate so don't use a number too low
if you already have acceptable FPS (around 30, can be seen on the up-left corner when the algorithm is running)

### If you want to use webcam and not RTSP

Use the following command instead: 

         demo_live.exe -uwc true 

If you need faster output: 

         demo_live.exe -uwc true -sc 0.5


         