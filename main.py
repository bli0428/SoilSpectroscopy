from qs_files.configs import Configs

# Initialize a Configs object with the path to your .qs file
config = Configs('path/to/your/config.qs')

# Load the configuration from the file
config.load()

# Access a specific section and key
value = config.get_value('section_name', 'key_name')

# Modify a value
config.set_value('section_name', 'key_name', 'new_value')

# Save the changes back to the file
config.save()