import json
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_name', type=str, default=None,
                        help='the name of the json file(s) from Cell Lcoator'
                        ' which you would like to convert. If json_name is a directory'
                        ' all annotations will be combined into single output json')
    parser.add_argument('--out_name', type=str, default=None,
                        help='the name of the output json' )

    args = parser.parse_args()

    if args.out_name is None:
        raise RuntimeError('Must specify out_name')
    if args.json_name is None:
        raise RuntimeError('Must specify json_name')


    # 
    # first level json defaults
    #
    fljd = {}
    fljd["currentId"] = 1

    #
    # first level json copied from first input json file
    #
    flfj = {}
    flfj["referenceView"] = "ReferenceView"
    flfj["ontology"] = "Ontology"
    flfj["stepSize"] = "StepSize"
    flfj["cameraPosition"] = "CameraPosition"
    flfj["cameraViewUp"] = "CameraViewUp"

    #
    # third level json defaults
    #
    tljd = {}
    tljd["type"] = "ClosedCurve"
    tljd["coordinateSystem"] = "LPS"

    # 
    # second level json maping
    #
    sljm = {}
    sljm["orientation"] = "SplineOrientation"
    sljm["representationType"] = "RepresentationType"
    sljm["thickness"] = "Thickness"


    #
    # initialize output json
    #
    converted = {}

    # create first level elements with default values
    for k,v in fljd.items() :
        converted[k] = v

    #
    # loop through input json files
    #
    json_file_list = []
    if os.path.isfile(args.json_name):
        json_file_list.append(args.json_name)
    elif os.path.isdir(args.json_name):
        file_name_list = os.listdir(args.json_name)
        for name in file_name_list:
            if name.endswith('json'):
                json_file_list.append(os.path.join(args.json_name, name))


    first_json = True
    converted["markups"] = []
    
    for findex, json_name in enumerate(json_file_list) :

        with open(json_name, 'rb') as in_file :
            annotation = json.load(in_file)
            markup = annotation['Markups'][0]

        if markup is None:
            continue

        #
        # If this is first file, create first level json elements
        #
        if first_json :

            for k,v in flfj.items() :
                converted[k] = markup[v]

            first_json = False

        #
        # initialize 
        #
        second = {}

        # 
        # name annotation by json filename
        #
        annotation_name = os.path.splitext(os.path.basename(json_name))[0]
        second["name"] = annotation_name

        # create second level elements with copied values
        for k,v in sljm.items() :
            second[k] = markup[v]

        # create third level
        second["markup"] = {}

        # create third level elements with default values
        for k,v in tljd.items() :
            second["markup"][k] = v

        # process control points
        second["markup"]["controlPoints"] = []

        for pindex, oldpt in enumerate(markup["Points"]) :

            newpt = {}
            newpt["id"] = str(pindex)
            newpt["position"] = [0,0,0]
            newpt["position"][0] = -1.0 * oldpt["x"]
            newpt["position"][1] = -1.0 * oldpt["y"]
            newpt["position"][2] =  1.0 * oldpt["z"] 

            newpt["orientation"] = [-1.0,-0.0,-0.0,-0.0,-1.0,-0.0,0.0,0.0,1.0] 

            second["markup"]["controlPoints"].append(newpt)          

        converted["markups"].append(second)

    #
    # write out json to file
    #
    with open( args.out_name, 'w') as out_file:
        json.dump(converted, out_file, indent=4)

    



