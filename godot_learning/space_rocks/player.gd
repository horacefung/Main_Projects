extends RigidBody2D
@export var engine_power = 500
@export var spin_power = 8000

var thrust = Vector2.ZERO
var rotation_dir = 0
var screensize = Vector2.ZERO

# behind the scenes, enum sets varibles as 0,1,2,3
enum {INIT, ALIVE, INVULNERABLE, DEAD} 
var state = INIT

# Called when the node enters the scene tree for the first time.
func _ready():
	change_state(ALIVE)
	screensize = get_viewport_rect().size

# Create a state change function
func change_state(new_state):
	# If input new state matches any of these
	# Set true/false for disabled parameter of CollisionShape2D
	match new_state:
		INIT:
			$CollisionShape2D.set_deferred("disabled", true)
		ALIVE:
			$CollisionShape2D.set_deferred("disabled", false)
		INVULNERABLE:
			$CollisionShape2D.set_deferred("disabled", true)
		DEAD:
			$CollisionShape2D.set_deferred("disabled", true)
	
	state = new_state

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	get_input()
	
func get_input():
	thrust = Vector2.ZERO
	if state in [DEAD, INIT]:
		return
	if Input.is_action_pressed("thrust"):
		# Transform x maintains the vector of body's forward direction?
		thrust = transform.x * engine_power
		# Get rotational vector (clockwise, counterclockwise, zero)
		# depending on inputs of left and right
		
	rotation_dir = Input.get_axis("rotate_left", "rotate_right")
	
# the convetion for defining processes around a physics body is _physics_process
func _physics_process(delta):
	# The get_input function will fetch user inputs and calculate variables
	# here will actually apply it to the body. By default, RigidBody2D will
	# search for _physics_process in its script.
	constant_force = thrust
	constant_torque = rotation_dir * spin_power

# By default, rigidt body looks for _integrate_forces if you want to adjust
# physics. Otherwise, forcing the position to change will mess up the physics
# calculations.
func _integrate_forces(physics_state):
	var xform = physics_state.transform
	xform.origin.x = wrapf(xform.origin.x, 0, screensize.x)
	xform.origin.y = wrapf(xform.origin.y, 0, screensize.y)
	physics_state.transform = xform
