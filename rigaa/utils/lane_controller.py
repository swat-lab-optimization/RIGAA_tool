
import math 
import numpy as np
class LaneController:
    def __init__(self, waypoints, speed):
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.done = False
        self.max_steering = math.pi
        #self.cutoff_frequency = 3
        self.previous_yaw = 0
        self.window = 10
        self.point_limit = 10
        self.speed_increment = 1
        self.max_speed = 30
        self.speed = speed
        self.min_speed = 8

    def control(self, x, y, yaw, speed):
        # Find the next waypoint
        closest_distance = float('inf')
        closest_waypoint = self.current_waypoint
        
        for i, waypoint in enumerate(self.waypoints[self.current_waypoint:self.point_limit]):
            self.point_limit = min(len(self.waypoints)-1,self.current_waypoint + self.window)
            distance = math.sqrt((x - waypoint[0])**2 + (y - waypoint[1])**2)
            if distance < closest_distance:
                closest_distance = distance
                #i = min(i, 10)
                closest_waypoint = i + self.current_waypoint
        self.current_waypoint = closest_waypoint
        #print(self.current_waypoint)
        #print("Waypoint:", self.current_waypoint)



        # Calculate the target yaw based on the waypoint
        if self.current_waypoint >=  len(self.waypoints) - 4:
            self.done = True
            steering = 0
            speeed = 0
        else:



            
            dx = self.waypoints[self.current_waypoint+1][0] - x
            dy = self.waypoints[self.current_waypoint+1][1] - y
            
            target_yaw = math.atan2(dy, dx)
            if dy > 0:# and dx < 0:
                target_yaw -= 2*math.pi
            
            self.previous_yaw = target_yaw

            # Calculate the steering angle
            steering = target_yaw - yaw

            self.speed = speed
            
            
            if abs(target_yaw - yaw) < 0.4: # if the vehicle is going straight
                self.speed += self.speed_increment
                if self.speed > self.max_speed:
                    self.speed = self.max_speed
            elif abs(target_yaw - yaw) > 1.2:
                self.speed -= self.speed_increment/2
                if self.speed <  self.min_speed:
                    self.speed = self.min_speed


            # Limit the steering angle
            
            if steering > math.pi:
                steering = steering - 2*math.pi
            elif steering < -math.pi:
                steering = steering + 2*math.pi

            #if (steering - prev_steering) >= 2.5:
            #    steering = prev_steering
            #elif (steering - prev_steering) < -2.5:
            #    steering = prev_steering

            #dt = 0.1
            
            #alpha = dt / (1.0/self.cutoff_frequency + dt)
            #steering = alpha * steering + (1.0 - alpha) * self.previous_steering
            #self.previous_steering = steering
            


            steering = min(steering, self.max_steering)
            steering = max(steering, -self.max_steering)
            #print("Steering:", steering)





        return steering, self.speed, closest_distance, self.done


    
    def get_angle(self, node_a, node_b):
        """
        It takes two points, and returns the angle between them

        Args:
          node_a: The first node
          node_b: the node that is being rotated

        Returns:
          The angle between the two nodes.
        """
        vector = np.array(node_b) - np.array(node_a)
        cos = vector[0] / (np.linalg.norm(vector))

        angle = (math.acos(cos))

        if node_a[1] > node_b[1]:
            return -angle
        else:
            return angle

class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_steering = math.pi
        self.previous_error = 0.0
        self.integral = 0.0

    def control(self, target, current, dt):
        # calculate error
        error = target - current

        # calculate integral
        self.integral += error * dt

        # calculate derivative
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        # calculate control signal
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        control = min(control, self.max_steering)
        control = max(control, -self.max_steering)

        return control
