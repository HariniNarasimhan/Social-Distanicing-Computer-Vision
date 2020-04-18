# Social-Distanicing-Computer-Vision
To ensure social distance among people via surveillance cameras. Let's fight against Covid-19.

Computer vision plays its role in various domain especially with surveillance. Due to the coronavirus pandemic, the government and people ensure to maintain social distance among the citizens.
All over the country police and other officials physically check on the streets and roads to create an awareness among the people who fail to maintain the social distance. It is very tough to monitor each and every individual. Also the ratio between citizens of the country and the police forces is not pretty well balanced.

In this scenario, computer vision techniques can monitor the people through the surveillance cameras and can also alert the police forces to prioritize their physical presence based on the lack of social distancing in a particular area.
 
Pedestrian and person detection is widely used for various applications in research such as
1. Person re-identification and tracking
2. Traffic safety
3. Suspicious behavior detection`.
and many more in the domain of surveillance.
 
Person detection using YOLO detects the person and draws a bounding box around them giving the top left and bottom right coordinates of the boxes as shown here.
Using the coordinates extracted, the pixel distance between the two persons can be estimated. To get the depth information between persons is not just achieved by calculating a norm distance or euclidean distance between coordinates of the boxes. 
While calculating distance, depth information by considering the referred object (targeted person)as one side of a cuboid and other objects (other person in the area) as the other side of the cuboid. For each pair, three different distances are calculated as the depth differs on every side of the cuboid (due to inequality on the sides of the cuboid).
 
Another way of estimating the depth information monocular depth algorithm is used based on training the KITTI dataset. This algorithm is built by referring the solution in below link

https://github.com/nianticlabs/monodepth2/

A sample information of depth is given by the below image. The person with same color code are in same depth and different color code are farway
If a person has not maintained the social distance even with one another person is tagged as “NO SD” (No Social Distance). This algorithm has tested only with long view cameras and has proved to work well.
 

### Social Impacts:
 
1. Can avoid the physical presence of police forces in much less crowded areas.
2. If suddenly an area gets crowded or more than unacceptable number of people do not maintain social distance, then it can be alerted to control the situation.
3. Real time alert system can create an awareness among people to follow the rules.
4. Stampede prevention
 
Further improvisation can be employed to scale the implementation between all the CCTV cameras in a region. And can provide the combined result of a particular region

### To run:
1. Yo need a trained model to detect depth information that I have trained based on KITT dataset (with guidance in https://github.com/nianticlabs/monodepth2/) -(I have trained te model, you can also use the repository by nianticlabs for a trained model)
2. With a test video you can run the file 
```
python test_social_distance.py --video_path test_video.mp4 
```
Sample video tested on two videos as 
<p align="center">
  <img src="assets/video.gif" alt="example input output gif" width="600" />
</p>
