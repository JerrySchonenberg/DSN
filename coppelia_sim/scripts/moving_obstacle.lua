--defines a moving obstacle, will follow a predetermined path

--simRemoteApi.start(19999)

path=sim.getObjectHandle('Path#0')          --SET path name
object=sim.getObjectHandle('haunted_box#0') --SET object name
pathLength=sim.getPathLength(path)
posOnPath=0
v=0.15 --SET velocity

while true do
    l=posOnPath/pathLength
    if (l > pathLength) then
        l = pathLength
    end
    
    position=sim.getPositionOnPath(path,l)
    orientation=sim.getOrientationOnPath(path, l)
    
    position[3]=0.15 --SET height
    sim.setObjectPosition(object, -1, position)
    
    posOnPath=posOnPath + v*sim.getSimulationTimeStep()
    
    sim.switchThread()
end
