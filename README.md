# Local LLM Hosting and Evaluation System

A Python-based desktop application for locally hosting, testing, benchmarking, and comparing Large Language Models (LLMs) using Ollama — without relying on cloud-based AI APIs.

## Overview

The **Local LLM Hosting and Evaluation System** is a desktop application developed as part of the **EF101-P16 academic project at NED University of Engineering and Technology**.

The project explores how Large Language Models can be executed locally and evaluated under practical hardware constraints. It integrates locally hosted models into a Python application, generates responses to a standardized set of prompts, records performance measurements, and evaluates response quality.

The system provides a centralized interface for interacting with multiple models and collecting structured experimental results, making it easier to compare their performance on the same tasks.

## Objectives

- Host and interact with LLMs locally using Ollama.
- Integrate multiple locally available language models into a single Python application.
- Evaluate models using a fixed set of 10 prompts covering different types of tasks.
- Measure response generation time and available system resource usage.
- Compare model responses based on correctness, relevance, and completeness.
- Store conversations, model information, benchmark results, and evaluation data in structured files.
- Explore the practical trade-offs between model size, response quality, and computational performance.

## Key Features

- **Local model execution:** Run language models on the local machine through Ollama.
- **Multi-model integration:** Work with multiple models through a unified application.
- **Desktop interface:** Interact with the system through a graphical interface built with Python Tkinter.
- **Standardized benchmarking:** Test models against the same 10 predefined prompts.
- **Performance monitoring:** Record response times and available system resource measurements.
- **Response evaluation:** Compare generated answers using correctness, relevance, and completeness criteria.
- **Persistent data storage:** Save conversation history and experimental results in JSON and CSV files.
- **Structured reporting:** Organize collected results into files suitable for analysis and documentation.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic and integration |
| Tkinter | Desktop graphical user interface |
| Ollama | Local LLM hosting and inference |
| REST API / HTTP requests | Communication between Python and Ollama |
| JSON | Conversation history and structured data storage |
| CSV | Benchmark results, model tests, and evaluation records |
| psutil (optional) | System resource monitoring, where available |

## Language Models

The application is configured to work with the following models:

| Model | Parameter count | Purpose |
|---|---:|---|
| `gemma3:4b` | Approximately 4 billion | Local inference and benchmark evaluation |
| `llama3.2:3b` | Approximately 3 billion | Local inference and comparative evaluation |

Both models are accessed through Ollama and execute locally once downloaded.

Model availability, memory requirements, inference speed, and output quality depend on the machine, model version, quantization, and runtime configuration.

## System Architecture

The application follows a modular workflow:

1. **User interface:** Tkinter provides the interface for interacting with the application.
2. **Model integration:** Python sends requests to Ollama's local API.
3. **Inference:** Ollama loads the selected model and generates a response.
4. **Performance measurement:** The application records relevant timing and system resource information.
5. **Benchmarking:** The models are tested against a consistent set of predefined prompts.
6. **Evaluation and storage:** Responses and measurements are saved in structured JSON and CSV files for analysis.

### Workflow

```text
          User
           |
           v
    Tkinter Interface
           |
           v
    Python Application
           |
           v
     Ollama Local API
           |
      +----+----+
      |         |
      v         v
   Gemma 3   Llama 3.2
      |         |
      +----+----+
           |
           v
   Generated Responses
           |
           v
 Performance Measurement
    and Evaluation
           |
           v
   JSON and CSV Files
```

## Benchmarking Methodology

The system uses **10 predefined prompts (P01–P10)** to examine different model capabilities under a consistent testing framework.

The prompt set covers:

- **Technical knowledge:** Questions involving computer hardware, storage, and electrical concepts.
- **Programming:** Python function generation and syntax correction.
- **Summarization:** Condensing information into a shorter form.
- **Text transformation:** Rewriting content according to instructions.
- **Structured output:** Generating responses in JSON format.
- **Number systems:** Converting binary values.
- **Numerical problem-solving:** Applying electrical principles to calculate resistance or current.

### Evaluation Criteria

Model responses can be assessed using the following criteria:

| Criterion | Description |
|---|---|
| Correctness | Whether the response is factually or technically accurate |
| Relevance | Whether the response addresses the prompt |
| Completeness | Whether the response includes the important information required |
| Response time | How long the model takes to generate a response |
| Resource usage | Available measurements of CPU, memory, and other relevant system resources |

The three response-quality criteria help assess the usefulness of generated answers, while performance measurements provide insight into the computational cost of local inference.

Results should be interpreted in the context of the hardware, model configuration, and testing conditions used during the experiments.

## Project Files and Data

The application maintains separate files for its operational history, benchmark outputs, model information, and test results.

| File | Purpose |
|---|---|
| `main.py` | Main Python application and entry point |
| `history.json` | Saved application history |
| `benchmark_results.csv` | Detailed benchmark records |
| `load_model_tests.csv` | Records associated with model-loading tests |
| `conversation_log.csv` | Stored conversation records |
| `model_info.csv` | Model information and configuration records |
| `benchmark_summary.csv` | Aggregated benchmark results |
| `benchmark_summary_by_prompt.csv` | Benchmark summaries grouped by prompt |

*The exact files present in the repository depend on the version of the project being shared. Some data files may be generated or updated when the application runs.*

## Requirements

- Windows or another operating system supported by Python and Ollama.
- Python 3.10 or later recommended.
- Ollama installed and running.
- Sufficient RAM and available storage for the selected models.
- Internet access for initial software and model downloads; model inference itself can run locally afterward.

## Installation and Setup

### 1. Install Python

Download Python from the official website:

https://www.python.org/downloads/

Verify the installation:

```bash
python --version
```

### 2. Install Ollama

Download and install Ollama:

https://ollama.com/download

Verify that Ollama is available:

```bash
ollama --version
```

### 3. Download the models

Run the following commands in a terminal:

```bash
ollama pull gemma3:4b
ollama pull llama3.2:3b
```

These commands download the model files needed for local execution. They can require several gigabytes of storage.

### 4. Install Python dependencies

Open a terminal in the project directory and install the required package:

```bash
python -m pip install requests
```

If system resource monitoring is enabled in the application, install the optional dependency:

```bash
python -m pip install psutil
```

Tkinter is generally included with standard Windows Python installations.

### 5. Start Ollama

Ensure that Ollama is running before launching the application. On installations where the Ollama service is not started automatically, start it using:

```bash
ollama serve
```

If the service is already running, you do not need to start a second instance.

### 6. Run the application

Navigate to the directory containing `main.py` and execute:

```bash
python main.py
```

The desktop application should open, allowing you to interact with the configured models and use the available testing and benchmarking functionality.

## Privacy and Local Execution

A central objective of this project is to demonstrate local language-model inference rather than relying on external cloud AI APIs.

After the software and required model files have been downloaded, inference can be performed locally through Ollama. Application data can also be stored locally in JSON and CSV files.

Local execution does not automatically guarantee complete privacy: operating-system services, other software, and any additional network-enabled functionality may still communicate externally. The application should be reviewed to verify its actual network behavior.

## Limitations

- Inference speed depends on the available hardware and model configuration.
- Larger models may require substantial RAM and storage.
- Model responses may contain inaccurate or incomplete information.
- Performance measurements can vary between runs.
- Resource monitoring depends on the metrics exposed by the system and the monitoring implementation.
- Response-quality scores require a consistent evaluation rubric to support meaningful comparisons.
- Results from one machine may not generalize to other hardware configurations.

## Future Improvements

Potential extensions include:

- Adding support for more local language models.
- Introducing automated scoring and more reproducible evaluation procedures.
- Generating visual comparisons of response time and resource consumption.
- Improving experiment reproducibility through recorded model and hardware configurations.
- Exporting benchmark reports in additional formats.
- Enhancing the interface with model-selection controls and richer evaluation summaries.

## Academic Context

**Course:** EF101 — IT Fundamentals  
**Institution:** NED University of Engineering and Technology  
**Project:** P16 — Local LLM Hosting and Evaluation System

This project demonstrates practical integration of Python, local AI inference, graphical user interfaces, API communication, data logging, and experimental evaluation.


