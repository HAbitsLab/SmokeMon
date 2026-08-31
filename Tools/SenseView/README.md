# SenseView



## Installation

```sh
pip install requirements.txt
```


## Sample Usage

> Running an existing layout 
```sh
python SenseView.py -i [Path to file or folder] -l [Layout Name] 
```

> RGB View Example:

```sh
python SenseView.py -i "./ExampleData/RGB/" -l RGB_Layout  
```

> MLX View Example:
```sh
python SenseView.py -i "./ExampleData/MLX/MLX.csv" -l MLX_Layout  
```


> View with a settings (senseView.yaml)
```sh
python SenseView.py -i "./ExampleData/senseView.yaml" -l RGB_Layout 
```


## Creating a Layout

A layout can contain multiple widgets. For example, you might want to view 2 RGB widgets or 1 RGB and 1 MLX widget.

Follow the instructions [here](https://www.google.com) to create your own layout.

The current implemented layouts are found under the `./Layouts` folder.


## Creating Annotations

Currently the system allows for 1 level of annotations.

<kbd>A</kbd>: Move the cursor left 

<kbd>D</kbd>: Move the cursor right

<kbd>S</kbd>: Start or Stop annotations

  
