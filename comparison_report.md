# Comparison Report

## Prompt

The most important principle when designing a multi-agent system is

## Hyperparameters

- Learning Rate: 5e-05
- Batch Size: 4
- Epochs: 5

## Base Model Outputs

### Output 1
The most important principle when designing a multi-agent system is to make sure you have the right tools for each system, to make sure that you have the right tools for each system. For this reason, we recommend an easy way to build a multi-agent system which will give you the best possible tools for the user's needs. If you have a multi-agent system with multiple languages and a common language, you probably want to understand how you make your system work. If you have multiple languages, you may have to figure out which language will work for you, so you will not need to look any further than the standard library of your language.



Building a multi-agent system
The goal of building a multi-agent system is to provide a solution for your application. For example

### Output 2
The most important principle when designing a multi-agent system is that it is relatively simple: you have to make sure that the application is consistent and that your application has an application that has some consistency.



In this case, we have two methods for building a multi-agent system. The first is the one that we need to define a single single agent. This is the most important principle when designing a multi-agent system. In this case, we have two methods for building a multi-agent system. The first is the one that we need to define a single agent. This is the most important principle when designing a multi-agent system. In this case, we have two methods for building a multi-agent system. This is the most important principle when designing a multi-agent system

### Output 3
The most important principle when designing a multi-agent system is to ensure that the code is properly integrated with the appropriate code.

### Output 4
The most important principle when designing a multi-agent system is to avoid being an in-house "scrum" tool for developers and companies that are building proprietary software systems (for example, they have a proprietary tool that can be used to develop software for a company that uses a proprietary software platform). For example, a developer would find that there are thousands of other software systems that the developer would use, and that there have been thousands of other applications that were built on top of the existing software.




There are a few ways in which such a system may make use of a proprietary software platform, which includes some tools, software features, and features that have been made available through different different vendors. Such a system may allow for a variety of different things, but there are also many limitations

### Output 5
The most important principle when designing a multi-agent system is that you have to be very careful in choosing the one you want to build. This can be done as well as from the design. For example, if you use a single agent, you may want to use multiple agents that you can build separately, so that you could have separate agents that could be created in separate systems.





The one thing to keep in mind is that when building a multi-agent system, you need to have a consistent user interface with each agent. So if the user needs multiple agents you may want to build in separate systems, so that you can keep multiple agents that are similar and compatible with each other. This is a great way to build your own multi-agent system (and for both of

## Fine-Tuned Model Outputs

### Output 1
The most important principle when designing a multi-agent system is to produce a coherent set of schema fields, which can vary across agents across multiple branches and introduce additional complexities or problems.

Distributed schema fields must be represented as the representation of the agent's final content before it can be shared between agents. Systems where a single agent can contain a single "id" or a common identifier (such as a short-term credit score) are particularly vulnerable to malformed schema fields.

Section 1: Key Agents

Section 2: Key Agents

Each agent responsible for each component within a specific set of constraints must be represented as the same object: a single key agent, a small agent, or a specialized object.
Section 3: Key Agents

Each agent responsible for each component within

### Output 2
The most important principle when designing a multi-agent system is to carefully label a specific agent that can produce new responses for any given agent, and to label it if possible, rather than merely listing the specific agent's output rather than the exact version.

Retrieval for GraphQL
The graph model described below assumes the following:

a) Input flow is represented by a list of nodes and outputs: a list of nodes, a list of edges, a list of possible outputs, a list of the most recent subtask flags, a list of the most recent examples of node-level generation, a given set of labels, and a set of labels for any given generation steps. Each node contains its own input flow — each node includes its own output agent, an external pipeline containing outputs from

### Output 3
The most important principle when designing a multi-agent system is to avoid having to create a separate agent that delegates the full responsibility for the entire process. This is where multi-agent design helps avoid the need to separate the entire pipeline, with each agent taking responsibility for each part of the pipeline.

Section 12: Designing a Multi-Agent Service
A multi-agent service should represent a specific set of actions in the system and that is the primary reason why they should be used. A multi-agent service should represent a single agent object and a single tool, a task agent that processes, searches, analyzes, or performs a task. Each agent should have its own type of API object, a single agent node, a dedicated agent object, a small subset of agents with an independent set of

### Output 4
The most important principle when designing a multi-agent system is to distinguish between different agents for each agent. This distinction requires two distinct pipelines: one for database queries and the other for production logs. If an agent that's acting independently of the rest of the pipeline must act independently of each other, the pipeline that has the ability to handle the query is better served by distinguishing between different agents.

Section 14: Deploying Agents

Deploying agents from the external node in a given production pipeline can require managing multiple nodes, and for each agent whose data is being deployed, the deployment step should be the same as its intended use: if a node in the pipeline has a "distributed" version of the app, the corresponding version should be available for the rest of the pipeline. In addition to deploying

### Output 5
The most important principle when designing a multi-agent system is to identify the appropriate context that should be relevant for the agent and the model that should be relevant for it. In this case, the tool should identify a given context that should not depend entirely on the exact context.

Distributed systems use multi-agent pipelines, with a single downstream pipeline to represent all the incoming requests, rather than one that must maintain state. A single downstream pipeline, for instance, updates a system's system's request log every time it is read. This allows for more flexible retrieval, since a single batch of requests from downstream nodes can represent that request and request immediately.

Distributed systems use multiple routing layers, each representing a given domain, each representing a given model's state, each representing a given agent's

# Qualitative Comparison

## Style

The base DistilGPT-2 model generates fluent English text but remains
generic and broad in its explanations. The fine-tuned model adopts a
more technical writing style, closely resembling software architecture
documentation. It frequently uses terminology such as "pipeline",
"schema", "agent", "node", and "distributed systems", reflecting the
training corpus.

## Coherence

The base model produces grammatically correct sentences but often repeats
phrases and drifts toward unrelated concepts. The fine-tuned model
maintains better focus on the topic of multi-agent systems and generally
stays within the intended domain. However, occasional repetition remains,
likely due to the relatively small size of the training dataset.

## Domain Adherence

The most significant improvement is the increased use of domain-specific
concepts. The fine-tuned model references ideas such as schema
management, routing, pipelines, distributed systems, deployment,
retrieval, and agent responsibilities. These concepts are present in the
custom training corpus and rarely appear in the base model outputs.

## Strengths

- Better technical vocabulary
- Improved consistency with the dataset
- More documentation-like writing style
- Better understanding of multi-agent architectures

## Limitations

The custom dataset contains fewer tokens than typically recommended for
effective language model specialization. As a result, the model still
occasionally repeats words or sentence structures and sometimes produces
partially incomplete explanations.

## Conclusion

Fine-tuning successfully specialized the pretrained DistilGPT-2 model
toward the domain of multi-agent LLM systems. Although the model still
inherits some limitations of small-scale fine-tuning, it demonstrates
clear improvements in technical terminology, topic consistency, and
domain-specific writing style.

## Dataset

The model was fine-tuned on a custom technical corpus describing the
architecture of production-grade multi-agent LLM systems. The dataset
covers topics including:

- Router Agents
- Shared State Management
- Retrieval-Augmented Generation (RAG)
- Tool Calling
- Human-in-the-Loop
- Prompt Engineering
- Deployment
- Security
- Observability
- Evaluation

The corpus was preprocessed using the GPT-2 tokenizer before training.

## Training Results
[EPOCH 1/5] avg loss = 4.4115
[EPOCH 2/5] avg loss = 3.9220
[EPOCH 3/5] avg loss = 3.5901
[EPOCH 4/5] avg loss = 3.2998
[EPOCH 5/5] avg loss = 3.0547

## Base Model

The pretrained model used for this project is DistilGPT-2, loaded from
the Hugging Face Transformers library.