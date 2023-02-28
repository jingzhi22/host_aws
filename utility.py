from _ctypes import PyObj_FromPtr
import rhino3dm
import json
import re
import numpy as np
import pandas as pd
import datetime
import pickle
import shapely
import compute_rhino3d.Grasshopper as gh
import pareto


class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr

def FormatJsonObj(jsonObj, filename = None):
    for layer1key in jsonObj.keys():
        for layer2key in jsonObj[layer1key]:
            for layer3key in jsonObj[layer1key][layer2key]:
                jsonObj[layer1key][layer2key][layer3key] = NoIndent(jsonObj[layer1key][layer2key][layer3key])
    output = json.dumps(jsonObj, cls=MyEncoder, sort_keys=True, indent=2)
    if filename:
        with open(filename, 'w') as outfile:
            outfile.write(output)
    return output

def ResponseSummary(output,filename=None):
    ans = {}
    for value in output['values']:
        ans[value['ParamName']] = {}
        for branchKey in value['InnerTree'].keys():
            ans[value['ParamName']][branchKey] = len(value['InnerTree'][branchKey])
    for key,value in ans.items():
        print(key,': ',value)
    if filename:
        with open(filename, 'w') as outfile:
            json_object = json.dumps(output, indent=2)
            outfile.write(json_object)

def Write3DM(parcelJSON= None,roadJSON = None,buildingJSON=None,filename=None):

    model = rhino3dm.File3dm()
    parcels,roads,buildings = [],[],[]

    if parcelJSON:
        for id,feature in parcelJSON.items():
            coordinates = [rhino3dm.Point3d(c[0],c[1],c[2]) for c in feature['geometry']['coordinates']]
            try:
                polyline = rhino3dm.Polyline(coordinates)
                parcels.append(polyline)
                model.Objects.AddPolyline(polyline)
            except:
                print('error from parcels')
    if roadJSON:
        for id,feature in roadJSON.items():
            coordinates = [rhino3dm.Point3d(c[0],c[1],c[2]) for c in feature['geometry']['coordinates']]
            try:
                roads.append(rhino3dm.Line(coordinates[0],coordinates[1]))
                model.Objects.AddLine(coordinates[0],coordinates[1])
            except:
                print('error from roads')
    
    if buildingJSON:
        for id,feature in buildingJSON.items():
            height = feature['properties']['storeys'] * 3.5
            coordinates = [rhino3dm.Point3d(c[0],c[1],c[2]) for c in feature['geometry']['coordinates']]
            try:
                polyline = rhino3dm.Polyline(coordinates)
                curve = polyline.ToNurbsCurve()
                extrusion = rhino3dm.Extrusion().Create(curve,height,True)
                if extrusion.PathTangent.Z < 0:
                    extrusion = rhino3dm.Extrusion().Create(curve,-height,True)
                buildings.append(extrusion)
                model.Objects.AddExtrusion(extrusion)
            except:
                print('error from buildings')
            
            try:
                obstacleCoordinates = [rhino3dm.Point3d(c[0],c[1],c[2]) for c in feature['properties']['obstacle']]
                obstaclePolyline = rhino3dm.Polyline(obstacleCoordinates)
                model.Objects.AddPolyline(obstaclePolyline)
            except:
                print('error from obstacle')
    
    if filename:
        model.Write(filename,7)
    return

def FlattenParameters(nestList):
    ans = []
    for nest in nestList:
        for n in nest:
            ans.append(n)
    return ans

def NestParameters(flatList, number):
    ans = []
    temp = []
    for x in flatList:
        temp.append(x)
        if len(temp) == number:
            ans.append(temp)
            temp = []
    return ans

def GetAverage(list1):
    return sum(list1)/len(list1)

def ReadResult(res):
    ans = []
    columns = []
    for generation, population in enumerate(res.history):
        for index, individual in enumerate(res.history[generation].pop):
            ans.append([generation, index] + list(individual.X) + list(individual.F) + list(individual.G))
            columns = ['gen','pop'] + ['x'+str(i) for i in range(len(list(individual.X)))] + ['f'+str(i) for i in range(len(list(individual.F)))] + ['g'+str(i) for i in range(len(list(individual.G)))]
    df = pd.DataFrame(np.array(ans))
    df.columns = columns
    return df

def SaveResults(res,precinct):
    curr_dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    ReadResult(res).to_csv('results\PYMOO_{0}.csv'.format(curr_dt_string),index=False)
    with open('results\Precinct_{0}.pickle'.format(curr_dt_string),'wb') as f:
        pickle.dump(precinct,f,protocol=pickle.HIGHEST_PROTOCOL)

def SelectFeature(jsonObj, UD_ID):
    for feature in jsonObj['features']:
        if feature['properties']['UD_ID'] == UD_ID:
            return feature

def Remap(Value, OldMin, OldMax, NewMin = 0, NewMax=1):
    return NewMin + ((Value - OldMin) * (NewMax - NewMin) / (OldMax - OldMin))
    
def GetEdgeCategory(selected_ID,rn_gdf, lu_gdf):

    profile = lu_gdf.loc[lu_gdf['UD_ID'] == selected_ID].geometry.boundary.values[0]
    splittedProfile = [shapely.geometry.LineString([profile.coords[i], profile.coords[i+1]]) for i in range(len(profile.coords) - 1)]
    edge_category = []
    closest_point = []

    for edge in splittedProfile:
        mp = edge.centroid
        dist_gdf = rn_gdf[['geometry','RD_TYP_CD']]
        distance = []
        closestpoint = []
        for ls in rn_gdf.geometry:
            d = ls.project(mp)
            p = ls.interpolate(d)
            ls_cp = shapely.geometry.Point(list(p.coords)[0])
            closestpoint.append(ls_cp.coords)
            distance.append(mp.distance(ls_cp))

        dist_gdf['distance'] = distance
        dist_gdf['closestpoint'] = closestpoint
        dist_gdf = dist_gdf.sort_values(by=['distance'])

        for i,row in dist_gdf.iterrows():
            if row.RD_TYP_CD not in ['Imaginary Line','T-Junction','Slip Road','Cross Junction']:
                if row.distance > 20:
                    edge_category.append(6)
                elif row.RD_TYP_CD == "Major Arterials/Minor Arterials":
                    edge_category.append(3)
                elif row.RD_TYP_CD == "Local Collector/Primary Access":
                    edge_category.append(4)
                elif row.RD_TYP_CD == "Local Access":
                    edge_category.append(5)
                closest_point.append(row.closestpoint[0])
                break
            
    return edge_category,closest_point

def CreateGHComputeInputs(inputs):
    compute_inputs = []
    for key,value in inputs.items():
        tree = gh.DataTree(key)
        tree.Append([0],[value])
        compute_inputs.append(tree)
    return compute_inputs

def GetPareto(df,objectives):
    col_index = [list(df.columns).index(obj) for obj in objectives]
    table = [list(df.itertuples(False))]
    nondominated = pareto.eps_sort(table, col_index)
    ndf = pd.DataFrame(nondominated, columns = df.columns)
    return ndf

def EvaluateGrasshopper(filename, parameters, ids):
    ghInputs = {}
    for key,value in zip(ids,parameters):
        ghInputs["RH_IN:{0}".format(key)] = value
    response = gh.EvaluateDefinition(filename, CreateGHComputeInputs(ghInputs))
    output = {}
    for d in response['values']:
        output[d['ParamName']] = d['InnerTree']
    return output

def check_dominance(pts):
    # if pt score = 0, means pt is dominated in 0 dimensions, truly the best, but unlikley.
    # it pt score = 1, means pt is dominated in 1 dimension only
    scores = []
    for A in pts:
        dominated = []
        for B in pts:
            # if dominated = 0, means A completedly dominates B
            # if dominated = 1, means A dominates B in 1 dimension
            dominated.append(sum([a>b for a,b in zip(A,B)]))
        scores.append(max(dominated))
    return scores

def sort_pareto(pts, min_pcn, min_samples):
    
    ans = []
    recursion_pts = pts

    if len(pts)*min_pcn > min_samples:
        min_samples = int(len(pts)*min_pcn)

    while True:

        dominance = check_dominance(recursion_pts)
        extract_pts = [pt for d,pt in zip(dominance,recursion_pts) if d == min(dominance)]
        recursion_pts = [pt for d,pt in zip(dominance,recursion_pts) if d != min(dominance)]
        
        ans.append(sorted(extract_pts, key=lambda x: x[0]))

        num_sorted_points = len([item for sublist in ans for item in sublist])
        conditionA = all([len(pts) <= min_samples , num_sorted_points==len(pts)])
        conditionB = all([len(pts) > min_samples , num_sorted_points > min_samples])
        
        if any([conditionA,conditionB]):
            break
    return ans

def get_pareto_index(pareto_set, pts):
    return [np.array([pts.index(pareto_pt) for pareto_pt in pareto_pts]).tolist() for pareto_pts in pareto_set]


def sort_pareto_by_set(df_fitness, min_pcn, min_samples, df_unstaked):
    # drop duplicates from optimisation
    df = df_fitness.drop_duplicates()
    # convert df to nested list
    pts = df.to_numpy().tolist()
    # sort pts according to logic
    pareto_sorted_pts = sort_pareto(pts, min_pcn, min_samples)
    # get original index from sorted points
    pareto_sorted_index = get_pareto_index(pareto_sorted_pts, pts)
    # return solutions as nested dfs
    pareto_sorted_df = []
    for i,pareto_set in enumerate(pareto_sorted_index):
        row = [df.iloc[pareto_index] for pareto_index in pareto_set]
        combined_rows = pd.DataFrame(row)
        combined_rows['nD_set'] = i
        pareto_sorted_df.append(combined_rows)
    
    sorted_dfs = pd.concat(pareto_sorted_df)
    final_df = df_unstaked.iloc[sorted_dfs.index]
    final_df['nD_set'] = sorted_dfs['nD_set']
    return final_df