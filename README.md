# Obstacule-Avoidance-Reinforcmente-Learning


This project was inpirated in lukovicaleksa Master thesis (https://github.com/lukovicaleksa/autonomous-driving-turtlebot-with-reinforcement-learning), where it uses Renforcement learning for obstacule avoidance. In our case we aply obstacules avoidance in multiples robots, the robots drives though a simulated factory, which one must to get a packge in a random load place and delivery it on a random unload place. We use the ROS framework for two main activities, the first one to control the robots and get Lidars data. The second one to create a new mensage that represents each load and unload places, with the objective of warning all the robots that the place is busy for some other robot.

For the simulation part, the STAGE simulator was used for the sake of the simplicity. This simulator is very usefull because we can create the most diferents enviroments using only paint or some other similar software. For example we create the first enviroment(as shows in a figure bellow) to train only one robot in how to avoid obstacules.


<br/>
<p align="center">
  <figcaption align="center"><b>Diagrama de blocos LIV</b></figcaption>
  <br/>
  <img src="https://github.com/Jaquetti/Obstacule-Avoidance-Reinforcmente-Learning/blob/main/Enviroment/canvas.png" />
  <br/>
</p>

<br/>

Is it very ugly right?  But it serves as shown in the video: https://www.youtube.com/watch?v=ydLoVypdTb0&t=3s&ab_channel=PEDROFELIPEJAQUETTI 


* How does it work? 
  
     It takes the 270° of the Lidar and set apart in the middle resulting in 2 areas, which one of this zones are set apart in midle again, result in 4 parts. As shown in the following figure.
      
<p align="center">
 
<br/>
  <img src="https://github.com/Jaquetti/images_of_all_repositores/blob/main/zones_mb.PNG" />
  <br/>
</p>

<br/>      
    This 4 areas results in 6 possible states, given a state space of 2^6 possible combinations. But theres some of those states impossible to happend, for example the representation of this state space is a tupple like (right_side, left_side, right_side_zone0,  right_side_zone1, left_side_zone0,  left_side_zone1) but the state (1,0,0,0,1,1) does not exist (I know it was a sheet ideia, i will fix it soon), at least it works, the only problem with this is some blank lines in the Q-table. The agent can take 4 action, the first one is move forward, the second one is move forward with some angular velocity(clockwise), the third one is move foward with some angular velocity(anti-clockwise) and the last one is the stopped action, where there is neither angular or linear velocity. With that information we can figure out that our Q-table has a shape of 64x4. The most important part of the algorith is the reward function. In this case there is 6 rules of reward:
    * When the robot crashs- Penalty -100 (It hurts a lot, if the robot could feel somethin)
    * If the robot keep insist in the stopped action- Penalty -0.5
    * If takes the Forward action, it receave a reward because we want our robot allways moving forward, not crazily turning from a side to another 
    * It suffers a penalty of -0.5 if the past action was turn left and the current turn right or vice-versa(Prevents the case discribed before)
    * If the distânce between the most closest point in the past and the current closest point is positve, it reciave a reward because means that the robot is more far of the    obstacule now(All of those distances are obtained by Lidar) 
    * The last one takes in consideration the distance of the path, so if the robot avoid the obstacule and this avoidance result in a point more closes to the destiny, a       reward is received. 
    
    
   



