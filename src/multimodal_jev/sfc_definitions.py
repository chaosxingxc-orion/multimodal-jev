"""Extract published SFC function definitions from input text, never from gold."""
import json


def unique_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate definition key')
        result[key]=value
    return result


def definitions(inputs):
    text=inputs['messages'][0]['content']
    start='### Available Functions:\n';end='\n\n### Environment:\n';audio="\n\n### User's Audio Request:\n"
    if text.count(start)!=1 or text.count(end)!=1 or text.count(audio)!=1:
        raise ValueError('unambiguous official function/environment sections required')
    prefix,rest=text.split(start);functions_text,rest=rest.split(end);environment_text,suffix=rest.split(audio)
    if suffix!='<audio>':raise ValueError('original first audio marker required')
    funcs=json.loads(functions_text,object_pairs_hook=unique_pairs)
    environment=json.loads(environment_text,object_pairs_hook=unique_pairs)
    if not isinstance(funcs,list) or not funcs or not isinstance(environment,dict):
        raise ValueError('function list and environment object required')
    names=[]
    for f in funcs:
        if (not isinstance(f,dict) or not isinstance(f.get('name'),str) or not f['name']
                or not isinstance(f.get('description'),str) or not f['description']):
            raise ValueError('named and described function required')
        params=f.get('parameters')
        if not isinstance(params,dict) or params.get('type')!='object' or not isinstance(params.get('properties'),dict):
            raise ValueError('object parameter schema required')
        required=params.get('required',[])
        if (not isinstance(required,list) or any(not isinstance(k,str) for k in required)
                or len(set(required))!=len(required) or not set(required)<=set(params['properties'])):
            raise ValueError('valid required parameter names needed')
        if any(not isinstance(v,dict) for v in params['properties'].values()):
            raise ValueError('parameter definitions must be objects')
        names.append(f['name'])
    if len(names)!=len(set(names)):raise ValueError('duplicate function name')
    return dict(functions=funcs,environment=environment,instruction_prefix=prefix)
