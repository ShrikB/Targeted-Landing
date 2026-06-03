# Targeted Landing Project

This project is designed to analyze video frames, apply image processing techniques, and detect landing zones using semantic segmentation. The main components of the project are organized into a structured directory, with each file serving a specific purpose.

## Project Structure

```
targeted-landing-docker
├── src
│   ├── Simulated_Modular_Test_metric.py  # Main processing logic for video analysis and landing zone detection.
│   ├── Mask_Merge_Singular.py             # Processes semantic segmentation masks to create safe, unsafe, and potential landing area masks.
│   ├── Landing_Zone_Singular.py            # Identifies landing zones in images using adaptive smoothing and scoring.
│   ├── Class_to_Color.py                   # Analyzes semantic segmentation images to find unique colors and their pixel counts.
│   └── OneFormer_Inference_Image.py        # Contains functions for processing images with the OneFormer model for semantic segmentation.
├── model
│   └── README.md                           # Documentation for the model directory, explaining its purpose and usage.
├── inputs
│   └── README.md                           # Documentation for the inputs directory, detailing expected input files and formats.
├── outputs
│   └── README.md                           # Documentation for the outputs directory, explaining the generated output files.
├── requirements.txt                        # Lists Python dependencies required for the project.
├── Dockerfile                              # Instructions for building the Docker image, including setup and dependencies.
├── docker-compose.yml                      # Defines services, networks, and volumes for the Docker application.
└── README.md                               # Documentation for the project, including setup instructions and usage guidelines.
```

## Setup Instructions

1. **Clone the Repository**: 
   Clone this repository to your local machine using:
   ```
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory**:
   ```
   cd targeted-landing-docker
   ```

3. **Build the Docker Image**:
   Use the following command to build the Docker image:
   ```
   docker build -t targeted-landing .
   ```

4. **Run the Docker Container**:
   You can run the container using:
   ```
   docker run --rm -v $(pwd)/inputs:/app/inputs -v $(pwd)/outputs:/app/outputs targeted-landing
   ```

5. **Using Docker Compose**:
   If you prefer to use Docker Compose, you can start the services defined in `docker-compose.yml` with:
   ```
   docker-compose up
   ```

## Usage Guidelines

- Place your input video files in the `inputs` directory.
- Place your model in the `model` directory.
- If model has different class labeling, change merging labels in Mask_Merge_Singular.py
- Change respective labels in the Simulated_Modular_Test_metric.py file
- The processed output files will be saved in the `outputs` directory.
- Refer to the individual module README files for specific usage instructions related to each component.

## Additional Information

For any issues or contributions, please refer to the project's issue tracker or contact the maintainers.