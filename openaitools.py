import inspect
from pydantic import BaseModel, Field

class OpenAiTools:
    def __init__(self):
        self._tools = {}

    @property
    def tools(self):
        return [descr for name,(f,descr) in self._tools.items()]


    def __call__(self, f):
        self.register_tool_callback(f)
        return f

    def call_function(self, name, args:dict):
        f,descr = self._tools[name]
        signature = inspect.signature(f)
        # print(signature)
        kwargs = dict()
        for pname, param in signature.parameters.items():
            for pname, param in signature.parameters.items():
                # print(name,param)
                annotation = param.annotation
                # print(annotation)
                if not issubclass(annotation, BaseModel):
                    raise Exception(
                        "You must annotate parameters with classes derived from pydantic.BaseModel"
                    )
                pvalue = annotation.model_validate(args[pname])
                kwargs[pname] = pvalue
        return f(**kwargs)

    def register_tool_callback(self, f):
        name = f.__name__
        # print(name)
        signature = inspect.signature(f)
        # print(signature)
        properties = dict()
        for pname, param in signature.parameters.items():
            # print(name,param)
            annotation = param.annotation
            # print(annotation)
            if not issubclass(annotation, BaseModel):
                raise Exception(
                    "You must annotate parameters with classes derived from pydantic.BaseModel"
                )
            schema = annotation.model_json_schema()
            # print(schema)
            properties[pname] = schema
        self._tools[name] = f, (
            {
                "type": "function",
                "name": name,
                "description": f.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        )




