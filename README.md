# PTG Server-side Machine Learning

This repo is loaded by the server and any service defined in the docker-compose file will be brought up.

As a quickstart, you can just copy the entire `clip-zero` directory and make changes.

But doing it step-by-step, the basic checklist for setting up a new ML app:
 - create an app directory (e.g. `mkdir mymodel/`). This will be the context to build your docker image in.
 - create a Dockerfile in the app directory. You can copy from `clip-zero`
   - If you want to use the API, you can inherit `FROM ptgctl` and then `import ptgctl` in your code.
     - NOTE: it uses a python 3.9 base image (for pytorch compatibility, not included) and comes with opencv and Pillow installed for image processing
     - The `ptgctl` Docker image is built by the API server, however you can build it yourself by doing:
       ```bash
        git clone git@github.com:VIDA-NYU/ptgctl.git
        docker build -t ptgctl ./ptgctl
       ```
 - create a `requirements.txt` to define your python dependencies
 - add a service for your app in `/docker-compose.yaml` - (copy and rename one of the other ones)

Finally:
 - create your app at `main.py` (assuming you didn't change it in the Dockerfile)



## Example App Outline

> NOTE: This app example assumes that your model is dependent on
> an active recipe. If that is not the case, see the yolo example.

You can look in `clip-zero/main.py` for a full example. Here I'm just going through how I
might structure a streaming machine learning application.

The basic idea is:
 - load your model at initialization
 - then when a recipe is turned on we will
   - pull the recipe info and process the NLP
     - since the recipe text doesn't change, it'd be good to process this only once at the start
   - then inside a loop checking that the recipe is still enabled:
     - read frame
     - compute image features
     - compare with text
     - upload predictions
     - repeat...

I've separated out some of the other non-ML specific logic into the App base class so you can 
read through and change anything if it's not working for you.

```python
import ptgctl

# a generic base class to keep your model code cleaner
class App:
    api: ptgctl.API
    # reading and writing streams
    def read(self, stream_id, last_entry_id): ...
    def upload(self, streams_data): ...

    # script entrypoint
    @classmethod
    def main(cls, *a, **kw): ...


class MyApp(App):
    output_prefix = 'mymodel'
    def __init__(self, checkpoint_path, **kw):
        self.model = load_model(checkpoint_path)  # code to load your model

    def run_while_active(self, active_id, last='$'):
        # load and process recipe

        recipe = self.recipes.ls(recipe_id)
        # each is a list of strings - do some NLP or something!
        tools = recipe['tools']
        ingredients = recipe['ingredients']
        instructions = recipe['instructions']
        z_tools, z_ingredients, z_instructions = ...  # get text embedding

        # run while we're on the current recipe.
        while active_id == self.current_id:
            # read

            # read the next "main camera" image
            results = await self.read('main', last)
            # iterate over the results
            for sid, samples in results:
                for ts, data in samples:
                    # load

                    # always use the latest timestamp
                    last = ts
                    # load the image as a numpy array
                    image = holoframe.load(data[b'd'])['image']

                    # predict
                    
                    # compare image and text
                    z_image = ...  # get the image embedding
                    tools_pred = ...  # get the tools similarity
                    ingredients_pred = ...  # get the ingredients similarity
                    instructions_pred = ...  # get the instructions similarity

                    # upload
                    
                    # upload to 3 different streams under a common prefix. 
                    self.upload({
                        f'{output_prefix}:tools': dict(zip(tools, tools_pred)),
                        f'{output_prefix}:ingredients': dict(zip(ingredients, ingredients_pred)),
                        f'{output_prefix}:instructions': dict(zip(instructions, instructions_pred)),
                    })



if __name__ == '__main__':
    import fire
    fire.Fire(MyApp.main)
```