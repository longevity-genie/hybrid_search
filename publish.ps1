# Remove the 'dist' directory if it exists
Remove-Item -Recurse -Force dist -ErrorAction Ignore

# Create source and wheel distributions
python setup.py sdist bdist_wheel --universal

# Upload the package using Twine, specifying the .pypirc file one level up
twine upload --verbose dist/* --config-file (Resolve-Path "..\.pypirc")
