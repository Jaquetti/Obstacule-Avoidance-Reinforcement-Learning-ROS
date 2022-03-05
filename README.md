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

Is it very ugly right? But it serves as shown the video: https://www.youtube.com/watch?v=ydLoVypdTb0&t=3s&ab_channel=PEDROFELIPEJAQUETTI 


* How does it work? 
  
      It takes the 270° of the Lidar and set apart in the middle resulting in 2 areas, which one of this zones are set apart in midle again, result in 4 parts. As shown in the following figure.
      
      


<br/>
<p align="center">
  <figcaption align="center"><b>Diagrama de blocos LIV</b></figcaption>
  <br/>
  <img src="https://github.com/Jaquetti/images_of_all_repositores/blob/main/zones_mb.PNG" />
  <br/>
</p>

<br/>
