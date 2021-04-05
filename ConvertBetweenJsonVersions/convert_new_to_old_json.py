import json
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_name', type=str, default=None,
                        help='the name of the json file output from Cell Lcoator'
                        ' which you would like to convert')
    parser.add_argument('--out_name', type=str, default=None,
                        help='the name of the output directory you would like'
                        ' to store the output files')
    args = parser.parse_args()

    if args.out_name is None:
        raise RuntimeError('Must specify out_name')
    if args.json_name is None:
        raise RuntimeError('Must specify json_name')

    if not os.path.isfile(args.json_name):
        raise RuntimeError("Json file\n%s\ndoes not exist" % args.json_name)

    if not os.path.exists( args.out_name ) :
         os.makedirs( args.out_name )
    else :
        if os.path.isfile( args.out_name ) :
            raise RuntimeError("Output directory\n%s\nis a existing file" % args.out_name)

    #
    # first level json default
    #
    fljd = {}
    fljd["DefaultRepresentationType"] = "spline"
    fljd["DefaultStepSize"] = 25.0
    fljd["DefaultThickness"] = 50.0
    fljd["Locked"] = 0
    fljd["MarkupLabelFormat"] = "%N-%d"
    fljd["Markups_Count"] = 1
    fljd["TextList"] = [None]
    fljd["TextList_Count"] = 0

    #
    # first level json mappings
    #
    fljm = {}
    fljm["DefaultCameraPosition"] = "cameraPosition"
    fljm["DefaultCameraViewUp"] = "cameraViewUp"
    fljm["DefaultOntology"] = "ontology"
    fljm["DefaultReferenceView"] = "referenceView"

    #
    # first level to nested mappings
    #
    flnm = {}
    flnm["DefaultSplineOrientation"] = "orientation"

    #
    # second level json defaults
    #
    sljd = {}
    sljd["AssociatedNodeID"] = "unknown"
    sljd["Closed"] = 1
    sljd["Description"] = "unknown"
    sljd["ID"] = "unknown"
    sljd["Label"] = "Annotation-1"
    sljd["Locked"] = 1
    sljd["OrientationWXYZ"] = [0.0,0.0,0.0,1.0]
    sljd["Selected"] = 1
    sljd["StepSize"] = 25.0
    sljd["Visibility"] = 1

    #
    # second level json mappings
    #
    sljm = {}
    sljm["RepresentationType"] = "representationType"
    sljm["SplineOrientation"] = "orientation"
    sljm["Thickness"] = "thickness"

    #
    # second level to outer mapping
    #
    slom = {}
    slom["CameraPosition"] = "cameraPosition"
    slom["CameraViewUp"] = "cameraViewUp"
    slom["Ontology"] = "ontology"
    slom["ReferenceView"] = "referenceView"


    with open(args.json_name, 'rb') as in_file:

        annotation = json.load(in_file)
        markups = annotation['markups']
        
        if markups is not None :

            for annotation_index in range(len(markups)) :
                
                markup = markups[annotation_index]
                print( "processing: %s " % markup['name'] )

                #
                # initialize empty dictionary
                #
                converted = {}

                # create first level elements with default values
                for k,v in fljd.items() :
                    converted[k] = v

                # create first level elements with copied values
                for k,v in fljm.items() :
                    converted[k] = annotation[v]

                # create first level elements with nested values
                for k,v in flnm.items() :
                    converted[k] = markup[v]

                #
                # create Markup array
                #
                converted["Markups"] = []
                second = {}
                converted["Markups"].append(second)

                # create second level elements with default 
                for k,v in sljd.items() :
                    second[k] = v

                # create second level elements with copied values
                for k,v in sljm.items() :
                    second[k] = markup[v]

                # create second level elements with outer values
                for k,v in slom.items() :
                    second[k] = annotation[v]

                #
                # create Points array
                #
                second["Points"] = []
                second["Points_Count"] = str(len(markup["markup"]["controlPoints"]))

                for cp in markup["markup"]["controlPoints"] :

                    pt = {}
                    pt["x"] = -1.0 * cp["position"][0]
                    pt["y"] = -1.0 * cp["position"][1]
                    pt["z"] =  1.0 * cp["position"][2]

                    second["Points"].append(pt)

                #
                # Write json for this annotation to the output directory
                #
                output_path = os.path.join(args.out_name,"annotation_%d.json" % annotation_index )

                with open( output_path, 'w') as out_file:
                    json.dump(converted, out_file, indent=4)



