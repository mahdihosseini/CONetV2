
#adjacency list generation for Darts
def DartsAdj(model):
    adjList = {}

    # get the process indexes of each cell

    preProcessIdx = []
    tempProcessIdx = []
    convIdx = []
    lastCellNum = 0
    idx = 0
    for (name, param) in model.named_parameters():
        if len(param.shape) >= 2:
            if 'preprocess' in name:

                # new cell number

                if name[name.find('.') + 1] != lastCellNum:
                    lastCellNum = name[name.find('.') + 1]
                    if tempProcessIdx:
                        preProcessIdx.append(tempProcessIdx)
                    tempProcessIdx = []

                tempProcessIdx.append(idx)
            if(param.shape[1] != 1):
                convIdx.append(idx)
        idx += 1
    preProcessIdx.append(tempProcessIdx)

  
    for idx in convIdx:
        adjList[idx] = []

    # attach the stem to the first two cells

    adjList[convIdx[0]].append(preProcessIdx[0][0])
    adjList[convIdx[0]].append(preProcessIdx[0][1])
    adjList[convIdx[0]].append(preProcessIdx[1][0])

    # connect the sequential sep convs together, and get the first and last idx of each seq

    idx = 0
    lastSeqNum = 0
    lastCellNum = 0
    tempSeq = []
    firstLast = []
    cellFirstLast = []

    for (name, param) in model.named_parameters():
        if idx == 0:
            idx += 1
            continue

        if len(param.shape) == 4 and param.shape[1] != 1:

            # new sequence

            if lastSeqNum != name[name.find('o') + 4]:
                if tempSeq:
                    cellFirstLast.append([tempSeq[0],
                            tempSeq[len(tempSeq) - 1]])
                for i in range(len(tempSeq) - 1):
                    adjList[tempSeq[i]].append(tempSeq[i + 1])
                lastSeqNum = name[name.find('o') + 4]
                tempSeq = []

                    # new cell number

            if name[name.find('.') + 1] != lastCellNum:
                if cellFirstLast:
                    firstLast.append(cellFirstLast)
                else:
                    firstLast.append([])
                cellFirstLast = []
                lastCellNum = name[name.find('.') + 1]

            if 'preprocess' not in name:
                tempSeq.append(idx)

        idx += 1

    cellFirstLast.append([tempSeq[0], tempSeq[len(tempSeq) - 1]])
    for i in range(len(tempSeq) - 1):
        adjList[tempSeq[i]].append(tempSeq[i + 1])
    firstLast.append(cellFirstLast)
    firstLast.pop(0)

   # somewhat hard coded connections from pre to sep

    for i in range(len(preProcessIdx)):

        # if the cell has sep connections at all

        if firstLast[i]:

            # seps 0 2 connecting to pre 0............

            adjList[preProcessIdx[i][0]].append(firstLast[i][0][0])
            adjList[preProcessIdx[i][0]].append(firstLast[i][2][0])

            if len(preProcessIdx[i]) == 2:

                # seps 1 3 4 connecting to pre 1

                adjList[preProcessIdx[i][1]].append(firstLast[i][1][0])
                adjList[preProcessIdx[i][1]].append(firstLast[i][3][0])
                adjList[preProcessIdx[i][1]].append(firstLast[i][4][0])
            else:

                # seps 0 2 connecting to pre 1........

                adjList[preProcessIdx[i][1]].append(firstLast[i][0][0])
                adjList[preProcessIdx[i][1]].append(firstLast[i][2][0])

                # seps 1 3 4 connecting to pre 2

                adjList[preProcessIdx[i][2]].append(firstLast[i][1][0])
                adjList[preProcessIdx[i][2]].append(firstLast[i][3][0])
                adjList[preProcessIdx[i][2]].append(firstLast[i][4][0])

            # dil connecting to seps 0 1

            adjList[firstLast[i][0][1]].append(firstLast[i][5][0])
            adjList[firstLast[i][1][1]].append(firstLast[i][5][0])

            # early cells

            if i <= len(preProcessIdx) - 3:

                # next next cell

                for j in range(0, 6):
                    adjList[firstLast[i][j][1]].append(preProcessIdx[i
                            + 2][0])

                if len(preProcessIdx[i + 2]) == 3:
                    for j in range(0, 6):
                        adjList[firstLast[i][j][1]].append(preProcessIdx[i
                                + 2][1])

                # next cell

                if len(preProcessIdx[i + 1]) == 2:
                    for j in range(0, 6):
                        adjList[firstLast[i][j][1]].append(preProcessIdx[i
                                + 1][1])
                else:
                    for j in range(0, 6):
                        adjList[firstLast[i][j][1]].append(preProcessIdx[i
                                + 1][2])
            elif i == len(preProcessIdx) - 2:

            # 2nd last cell
                # next cell

                if len(preProcessIdx[i + 1]) == 2:
                    for j in range(0, 6):
                        adjList[firstLast[i][j][1]].append(preProcessIdx[i
                                + 1][1])
                else:
                    for j in range(0, 6):
                        adjList[firstLast[i][j][1]].append(preProcessIdx[i                    
                             + 1][2])
            #last cell
            else:
                for j in range(0, 6):
                    adjList[firstLast[i][j][1]].append(convIdx[len(convIdx)-1])
        else:

        # not too sure about cell 2 and 4, directly connect preprocess to next cells'?
            # early cells

            if i <= len(preProcessIdx) - 3:

                # next next cell

                adjList[preProcessIdx[i][0]].append(preProcessIdx[i
                        + 2][0])
                adjList[preProcessIdx[i][1]].append(preProcessIdx[i
                        + 2][0])

                if len(preProcessIdx[i + 2]) == 3:
                    adjList[preProcessIdx[i][0]].append(preProcessIdx[i
                            + 2][1])
                    adjList[preProcessIdx[i][1]].append(preProcessIdx[i
                            + 2][1])

                # next cell

                if len(preProcessIdx[i + 1]) == 2:
                    adjList[preProcessIdx[i][0]].append(preProcessIdx[i
                            + 1][1])
                    adjList[preProcessIdx[i][1]].append(preProcessIdx[i
                            + 1][1])
                else:
                    adjList[preProcessIdx[i][0]].append(preProcessIdx[i
                            + 1][2])
                    adjList[preProcessIdx[i][1]].append(preProcessIdx[i
                            + 1][2])
            elif i == len(preProcessIdx) - 2:

            # 2nd last cell
                # next cell

                if len(preProcessIdx[i + 1]) == 2:
                    adjList[preProcessIdx[i][0]].append(preProcessIdx[i
                            + 1][1])
                    adjList[preProcessIdx[i][1]].append(preProcessIdx[i
                            + 1][1])
                else:
                    adjList[preProcessIdx[i][0]].append(preProcessIdx[i
                            + 1][2])
                    adjList[preProcessIdx[i][1]].append(preProcessIdx[i
                            + 1][2])

    return adjList, [264]


def ResNetAdj(model):
	convIdx = []
	downIdx = set()
	adjList = {}
	idx = 0
	for name, param in model.named_parameters():
		if(len(param.shape) == 4 or len(param.shape) == 2):
			convIdx.append(idx)
			if("shortcut" in name):
				downIdx.add(idx)
		idx += 1

	idx = 0
	while(idx < len(convIdx)-1):
		if(convIdx[idx+1] in downIdx):
			adjList[convIdx[idx]] = [convIdx[idx+2]]
			adjList[convIdx[idx+1]] = [convIdx[idx+2]]
			idx += 2
		else:
			adjList[convIdx[idx]] = [convIdx[idx+1]]
			idx += 1
		
	idx = 0
	inc = 3
	nextinc = 3
	while(idx < len(convIdx)-3):
		inc = nextinc
		adjList[convIdx[idx]].append(convIdx[idx+inc])
		if((convIdx[idx+inc]) in downIdx):
			nextinc = 4
		else:
			nextinc = 3
		idx += inc-1
	return adjList, convIdx

def ResNextAdj(model):
	convIdx = []
	downIdx = set()
	adjList = {}
	idx = 0
	for name, param in model.named_parameters():
		if(len(param.shape) == 4):
			convIdx.append(idx)
			if("downsample" in name):
				downIdx.add(idx)
		idx += 1

	idx = 0
	while(idx < len(convIdx)-1):
		if(convIdx[idx+1] in downIdx):
			adjList[convIdx[idx]] = [convIdx[idx+2]]
			adjList[convIdx[idx+1]] = [convIdx[idx+2]]
			idx += 2
		else:
			adjList[convIdx[idx]] = [convIdx[idx+1]]
			idx += 1
		
	idx = 0
	inc = 3
	nextinc = 3
	while(idx < len(convIdx)-4):
		inc = nextinc
		adjList[convIdx[idx]].append(convIdx[idx+inc])
		if((convIdx[idx+inc]) in downIdx):
			nextinc = 5
		else:
			nextinc = 3
		idx += inc-1
	return adjList, convIdx




LAYER_SET = ["Conv2d", "Squeeze-Conv2d","Expand3-Conv2d","Expand1-Conv2d"]
# MainStream
LAYER_4D_SET = {"conv":LAYER_SET[0],
                "downsample":LAYER_SET[0],  #ResNet Pop
                "squeeze":LAYER_SET[1],     #SqueezeNet Pop
                
                
                }

# Stream 1
S1_LAYER_4D_SET = {"expand1":LAYER_SET[3]    #SqueezeNet Pop
				}

# Stream 2
S2_LAYER_4D_SET = {"expand3":LAYER_SET[2]      #SqueezeNet Pop
				}
				
def LayerDetection(string):
    for elem in (LAYER_4D_SET.keys()):
        if elem in string:
            return LAYER_4D_SET[elem]
    for elem in (S1_LAYER_4D_SET.keys()):
        if elem in string:
            return S1_LAYER_4D_SET[elem]
    
    for elem in (S2_LAYER_4D_SET.keys()):
        if elem in string:
            return S2_LAYER_4D_SET[elem]

    return LAYER_SET[0]



def MobileNetAdj(model):
	convIdx = []
	oneChannel = set()
	adjList = {}
	idx = 0
	for name, param in model.named_parameters():
		if(len(param.shape) >= 2):
			convIdx.append(idx)
			if(param.shape[1] == 1):
				oneChannel.add(idx)
		idx += 1

	idx = 0
	while(idx < len(convIdx)-1):
		if(convIdx[idx+1] in oneChannel):
			adjList[convIdx[idx]] = [convIdx[idx+2]]
			adjList[convIdx[idx+1]] = [convIdx[idx+2]]
			idx += 2
		else:
			adjList[convIdx[idx]] = [convIdx[idx+1]]
			idx += 1
	

	return adjList, convIdx


def MobileNetAdjForLL(model):
	convIdx = []
	oneChannel = set()
	adjList = {}
	idx = 0
	for name, param in model.named_parameters():
		if(len(param.shape) >= 2):
			convIdx.append(idx)
			if(param.shape[1] == 1):
				oneChannel.add(idx)
		idx += 1

	for i in range(len(convIdx)-1):
		adjList[convIdx[i]] = []

	idx = 0
	while(idx < len(convIdx)-1):
		if(convIdx[idx+1] in oneChannel):
			adjList[convIdx[idx]].append(convIdx[idx+1])
			adjList[convIdx[idx]].append(convIdx[idx+2])
			adjList[convIdx[idx+1]].append(convIdx[idx+2])
			idx += 2
		else:
			adjList[convIdx[idx]] = [convIdx[idx+1]]
			idx += 1

	return adjList, convIdx	




def SQNetAdj(model):
	adjList = {}
	Keys = {}
	for idx, (name, param) in enumerate(model.named_parameters()):
		if(len(param.shape) == 4):
			Name = LayerDetection(name)
			if(idx == 0): 
				adjList[idx] = []
				Keys[idx] = "main"
			
			elif(Name in LAYER_4D_SET.values()):
				if (idx <= 3):
					adjList[list(Keys.keys())[-1]].append(idx)
					adjList[idx] = []
					Keys[idx] = "main"
				else:
					adjList[list(Keys.keys())[-1]].append(idx)
					adjList[list(Keys.keys())[-2]].append(idx)
					adjList[idx] = []
					Keys[idx] = "main"

			elif (Name in S1_LAYER_4D_SET.values()):
				adjList[list(Keys.keys())[-1]].append(idx)
				adjList[idx] = []
				
				Keys[idx] ="S1"
				
			elif (Name in S2_LAYER_4D_SET.values()):
				adjList[list(Keys.keys())[-2]].append(idx)
				Keys[idx] = "S2"
				adjList[idx] = []

			
	# adjList[idx-1].append(idx-1)
				
	print(adjList)
	# print(list(Keys.keys()))
	return adjList, list(Keys.keys())


def SQNetAdjforHandCraft(model):
	adjList = {}
	Keys = {}
	Layers = []
	counter = 0
	for idx, (name, param) in enumerate(model.named_parameters()):
		print(name, param.shape)
		if((len(param.shape) == 4) or (len(param.shape) == 2)):
			Layers.append(idx)
			# print(name)
	for idx, (name, param) in enumerate(model.named_parameters()):
		if((len(param.shape) == 4) or (len(param.shape) == 2)):
			# Name = LayerDetection(name)
			print(name, idx)
			if(idx==0):
				adjList[idx] = [Layers[counter+1]]

			if(("fire" in name) and ("conv1" in name)): 
				adjList[idx] = [Layers[counter+1],Layers[counter+2]]
			
			
			elif(("fire" in name) and ("conv2" in name)): 
				adjList[idx] = [Layers[counter+2]]
			
			elif(("fire" in name) and ("conv3" in name)): 
				adjList[idx] = [Layers[counter+1]]
			
			
			counter +=1
		
			
	
	adjList[76] = []			
	return adjList, [76]
			
			

			
def ShuffleADJ(model):



	adjList = {0:[3,9], 3:[6], 6:[18, 27, 36, 45, 51], 9:[12], 12:[15], 15:[18,27, 36, 45, 51], 18:[21], 21:[24]
          ,24:[27, 36, 45, 51],27:[30], 30:[33], 33:[36, 45, 51], 36:[39], 39:[42],
		  
		  
		  42:[45,51], 45:[48],48:[60, 69, 78, 87, 96, 105, 114, 123, 129], 51:[54], 54:[57], 57:[60, 69, 78, 87, 96, 105, 114, 123, 129], 60:[63], 63:[66], 66:[69, 78, 87, 96, 105, 114, 123, 129]
		  , 69:[72], 72:[75], 75:[78,87, 96, 105, 114, 123, 129], 78:[81], 81:[84], 84:[87, 96, 105, 114, 123, 129], 87:[90], 90:[93], 93:[96, 105, 114, 123, 129]
		  , 96:[99], 99:[102], 102:[105, 114, 123, 129], 105:[108], 108:[111], 111:[114, 123, 129], 114:[117], 117:[120],
		  
		  
		  120:[123,129], 123:[126], 126:[138, 147, 156, 165], 129:[132], 132:[135], 135:[138, 147, 156, 165]
		  , 138:[141], 141:[144], 144:[147, 156, 165], 147:[150], 150:[153], 153:[156, 165], 156:[159], 159:[162],
		  
		  162:[165],
		  #FC Next
		  165:[]}


	return adjList, [165]

"""
def ShuffleAdj(model):

    adjList = {}
    convIdx = []
    idx = 0
    for (name, param) in model.named_parameters():
        if len(param.shape) == 4:
            convIdx.append(idx)
        idx += 1

    for idx in convIdx:
        adjList[idx] = []

    stage = -1
    substage = -1
    branch = -1
    tempSeq = []
    firstLast = []
    stageFirstLast = []
    stageFirstLastB1 = []
    stageFirstLastB2 = []
    
    idx = 0
    for (name, param) in model.named_parameters():
        if len(param.shape) == 4 and "branch" in name:

            #new branch
            if branch != name[name.find('h') + 1] or substage != name[name.find('.') + 1] :           
                if tempSeq:
                    if(branch == '1'):
                        stageFirstLastB1.append([tempSeq[0], tempSeq[len(tempSeq)-1]])
                    else:
                        stageFirstLastB2.append([tempSeq[0], tempSeq[len(tempSeq)-1]])
                    for i in range(len(tempSeq) - 1):
                        adjList[tempSeq[i]].append(tempSeq[i + 1])
                branch = name[name.find('h') + 1]
                substage = name[name.find('.') + 1]
                tempSeq = []
            tempSeq.append(idx)

            #new stage
            if stage != name[name.find('e') + 1]:
                if(stageFirstLastB1 and stageFirstLastB2):               
                    stageFirstLast.append(stageFirstLastB1)
                    stageFirstLast.append(stageFirstLastB2)
                    firstLast.append(stageFirstLast)
                    stageFirstLast = []
                    stageFirstLastB1 = []
                    stageFirstLastB2 = []
                stage = name[name.find('e') + 1]

        idx += 1

    for i in range(len(tempSeq) - 1):
        adjList[tempSeq[i]].append(tempSeq[i + 1])
    if(branch == '1'):
        stageFirstLastB1.append([tempSeq[0], tempSeq[len(tempSeq)-1]])
    else:
        stageFirstLastB2.append([tempSeq[0], tempSeq[len(tempSeq)-1]])

    if(stageFirstLastB1 and stageFirstLastB2):
        stageFirstLast.append(stageFirstLastB1)
        stageFirstLast.append(stageFirstLastB2)
        firstLast.append(stageFirstLast)

    lastStage = False

    #input to first stage
    adjList[convIdx[0]].append(firstLast[0][0][0][0])
    adjList[convIdx[0]].append(firstLast[0][1][0][0])
    
    for idx, stage in enumerate(firstLast):

        if(idx == len(firstLast) - 1):
            lastStage = True

        #intrastage connections
        branch2 = stage[1]
        branch1end = stage[0][0][1]

        for i in range(len(branch2)):
            j = i + 1
            if i != 0:
                adjList[branch1end].append(branch2[i][0])
            start = branch2[i][1]
            while j < len(branch2):
                adjList[start].append(branch2[j][0])
                j += 1
            #interstage connections
            if(not lastStage):
                adjList[start].append(firstLast[idx+1][0][0][0])
                adjList[start].append(firstLast[idx+1][1][0][0])
            else:
                adjList[start].append(convIdx[len(convIdx)-1])

        if(not lastStage):
            adjList[branch1end].append(firstLast[idx+1][0][0][0])
            adjList[branch1end].append(firstLast[idx+1][1][0][0])
        else:
            adjList[branch1end].append(convIdx[len(convIdx)-1])

  
    
    return adjList

"""






