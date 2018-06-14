import json
        
class JsonVariables(object):
    def __init__(self,variables_json_path,model__var_json_path):
        
        self.variables_json_path = variables_json_path            
        self.vars = json.load(open(self.variables_json_path))
                
        self.model__var_json_path = model__var_json_path    
        self.model_vars = json.load(open(self.model__var_json_path))