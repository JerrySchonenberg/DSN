--navigation program, which avoids obstacles using the Braitenberg algorithm
--originates from CoppeliaSim, tweaked for own purposes

function sysCall_init()
    count = 0
    visionSensorHandle = sim.getObjectHandle('Vision_sensor')

    usensors={-1,-1,-1,-1,-1,-1,-1,-1,-1}
    for i=1,9,1 do
        usensors[i]=sim.getObjectHandle("Pioneer_p3dx_ultrasonicSensor"..i)
    end
    motorLeft=sim.getObjectHandle("Pioneer_p3dx_leftMotor")
    motorRight=sim.getObjectHandle("Pioneer_p3dx_rightMotor")
    noDetectionDist=1.2
    maxDetectionDist=0.2
    detect={0,0,0,0,0,0,0,0,0}
    braitenbergR={-2.0, -1.6, -0.4, -0.2, 0.0, 0.0, 0.0, 0.0, -2.0}
    braitenbergL={0.0, 0.0, 0.0, 0.0, -0.2, -0.4, -1.6, -2.0, -2.0}
    v0=3
end

function sysCall_cleanup()

end

function sysCall_actuation()
    for i=1,9,1 do
        res,dist=sim.readProximitySensor(usensors[i])
        if (res>0) and (dist<noDetectionDist) then
            if (dist<maxDetectionDist) then
                dist=maxDetectionDist
            end
            detect[i]=1-((dist-maxDetectionDist)/(noDetectionDist-maxDetectionDist))
        else
            detect[i]=0
        end
    end

    vLeft=v0
    vRight=v0

    for i=1,9,1 do
        vLeft=vLeft+braitenbergL[i]*detect[i]
        vRight=vRight+braitenbergR[i]*detect[i]
    end

    count = count + 1
    if count == 2 then --Only every 2nd iteration
        -- Take screenshot of current view with vision sensor
        image = sim.getVisionSensorCharImage(visionSensorHandle, 0, 0, 256, 256, 0.99)
        baseDir = '/home/jerry/bachelor_thesis/coppelia_sim/dataset/'
        -- Directory is chosen based on vLeft and vRight
        if vRight >= vLeft + 1 then --Turn left
            filepath = baseDir..'left/l-'..os.date("%Y_%m_%d-%H_%M_%S",os.time())..'.png'
        elseif vLeft >= vRight + 1 then --Turn right
            filepath = baseDir..'right/r-'..os.date("%Y_%m_%d-%H_%M_%S",os.time())..'.png'
        else --Straight
            filepath = baseDir..'straight/s-'..os.date("%Y_%m_%d-%H_%M_%S",os.time())..'.png'
        end
        sim.saveImage(image, {256,256}, 1, filepath, -1) -- Save image
        count = 0
    end

    sim.setJointTargetVelocity(motorLeft,vLeft)
    sim.setJointTargetVelocity(motorRight,vRight)
end
