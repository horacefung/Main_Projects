extends Area2D

@export var speed = 1000

var velocity = Vector2.ZERO

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	position += velocity * delta

func start(_transform):
	# Pass in transform signal which will come from the ship/player
	# That way, the bullet always knows the correct position & rotation
	# to be pointing at
	transform = _transform
	velocity = transform.x * speed
	
func _on_visible_on_screen_notifier_2d_screen_exited():
	queue_free()

	
