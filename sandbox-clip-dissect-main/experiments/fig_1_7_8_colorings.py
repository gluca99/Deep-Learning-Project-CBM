def get_coloring(target_name, target_layer, neurons_to_display):
    
    if target_name == "resnet18_places" and target_layer=="layer4" and neurons_to_display=="CLIP-Dissect":
        def get_color(method, i):
            if method=="clip":
                if i in []:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="nd":
                if i in []:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="milan_b":
                if i in [0, 3, 5]:
                    return "orange"
                elif i in [2, 4, 6, 7, 8]:
                    return "red"
                else:
                    return "green"
            elif method=="milan_ood":
                if i in [5]:
                    return "orange"
                elif i in [0, 2, 3, 4, 6, 7, 8]:
                    return "red"
                else:
                    return "green"
                
    if target_name == "resnet18_places" and target_layer=="layer4" and neurons_to_display=="NetDissect":           
        def get_color(method, i):
            if method=="clip":
                if i in []:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="nd":
                if i in []:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="milan_b":
                if i in [0, 5]:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="milan_ood":
                if i in [0, 1, 3, 5, 6]:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
                
    if target_name == "resnet50" and target_layer=="layer4" and neurons_to_display=="NetDissect":
        def get_color(method, i):
            if method=="clip":
                if i in [2]:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="nd":
                if i in []:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="milan_b":
                if i in [0, 2, 4, 7]:
                    return "orange"
                elif i in [1, 5, 6, 8, 9]:
                    return "red"
                else:
                    return "green"
            elif method=="milan_ood":
                if i in [0, 6, 8]:
                    return "orange"
                elif i in [1, 2, 3, 4, 5, 7, 9]:
                    return "red"
                else:
                    return "green"
                
    if target_name == "resnet50" and target_layer=="layer4" and neurons_to_display=="CLIP-Dissect":
        def get_color(method, i):
            if method=="clip":
                if i in []:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="nd":
                if i in [2,6]:
                    return "orange"
                elif i in [3,4]:
                    return "red"
                else:
                    return "green"
            elif method=="milan_b":
                if i in [0, 2, 3, 4, 6, 8, 9]:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="milan_ood":
                if i in [3, 6, 8]:
                    return "orange"
                elif i in [0, 2, 4, 7, 9]:
                    return "red"
                else:
                    return "green"
    
    if target_name == "resnet50" and target_layer=="layer1" and neurons_to_display=="random":
        def get_color(method, i):
            if method=="clip":
                if i in []:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="nd":
                if i in []:
                    return "orange"
                elif i in [0]:
                    return "red"
                else:
                    return "green"
            elif method=="milan_b":
                if i in []:
                    return "orange"
                elif i in [0, 3]:
                    return "red"
                else:
                    return "green"
            elif method=="milan_ood":
                if i in []:
                    return "orange"
                elif i in [0, 3]:
                    return "red"
                else:
                    return "green"
    
    if target_name == "resnet50" and target_layer=="layer4" and neurons_to_display=="random":
        def get_color(method, i):
            if method=="clip":
                if i in []:
                    return "orange"
                elif i in []:
                    return "red"
                else:
                    return "green"
            elif method=="nd":
                if i in [0, 1]:
                    return "orange"
                elif i in [2]:
                    return "red"
                else:
                    return "green"
            elif method=="milan_b":
                if i in [2]:
                    return "orange"
                elif i in [0, 1]:
                    return "red"
                else:
                    return "green"
            elif method=="milan_ood":
                if i in [1]:
                    return "orange"
                elif i in [0, 2, 3]:
                    return "red"
                else:
                    return "green"
        
    return get_color