# TREVA Architecture

**TREVA** stands for **Typed-Relation Evidence-guided Visual Agent**. The name describes the intended research system; **SoftDoc** remains the parser-neutral document representation and `softdoc` remains the stable Python package and CLI name.

## End-to-end flow

```mermaid
flowchart TD
    subgraph OFFLINE[Offline document preparation]
        PDF[PDF] --> PARSER[MinerU parser backend]
        PARSER --> ADAPTER[MinerUAdapter<br/>raw parser conversion only]
        ADAPTER --> PIPE[SoftDocPipeline<br/>deterministic document passes]
        PIPE --> SD[SoftDoc<br/>Pages · Sections · Elements · Relations<br/>bbox · assets · provenance]
        SD --> SU[SearchUnitBuilder]
        SU --> IDX[BM25 + multilingual E5 indexes]
    end

    subgraph ONLINE[Online evidence-guided reading]
        ROOT[Root question] --> PLAN[Optional conservative Planner]
        PLAN --> TARGET[Current question and active evidence gap]
        TARGET --> EXACT[Exact anchor lookup]
        TARGET --> SEARCH[Hybrid retrieval]
        IDX --> SEARCH
        SEARCH --> SESSION[SearchSession<br/>resumable ranked candidates]
        SESSION --> PREVIEW[CandidatePreview batches]
        EXACT --> CTRL[Controller<br/>contract frozen; policy is next-stage work]
        PREVIEW --> CTRL

        CTRL --> READ[Read Element / Page / TableView]
        CTRL --> NAV[Follow confirmed Relation<br/>or inspect candidate navigation hint]
        CTRL --> INSPECT[Inspect visual source or region]
        SD --> READ
        SD --> NAV
        SD --> INSPECT

        READ --> READER[Text / Table / Visual Reader]
        NAV --> READER
        INSPECT --> READER
        READER --> OBS[ObservationStore<br/>grounded observations + limitations]
        OBS --> CHECK[Evidence Checker]
        MEM[EvidenceMemory] --> CHECK
        CHECK --> DELTA[Validated evidence delta]
        DELTA --> MEM
        MEM -->|incomplete: expose current gap| CTRL
        MEM -->|ready| ANSWER[Answerer]
        ANSWER --> OUTPUT[Answer + used evidence IDs<br/>program expands source citations]
    end

    SD -. navigation signal only .-> CTRL
    SD -. Relation is never Evidence .-> CHECK
```

## The safety boundary

```mermaid
flowchart LR
    LEAD[Search result / page position] --> READ[Actual read]
    CONF[Confirmed Relation] --> READ
    CAND[Candidate Relation<br/>navigation hypothesis only] --> READ
    READ --> OBS[Grounded Observation]
    OBS --> CHECK[Evidence Checker]
    CHECK -->|accepted| EVID[Evidence]
    CHECK -->|not useful or uncertain| KEEP[Keep read history;<br/>do not contaminate Evidence]
    EVID --> ANSWER[Answerer]
```

The intended invariants are:

- retrieval finds a place to start reading; it does not select final answer context;
- confirmed Relations are normal navigation handles;
- candidate Relations are investigation hints, not facts;
- a Relation, CandidatePreview, page location, or Reader limitation is never Evidence by itself;
- Readers produce Observations, and only the Checker may admit them into EvidenceMemory;
- the Answerer receives accepted Evidence rather than raw retrieval or navigation state;
- every accepted Evidence item remains traceable through its Observation and ReadRecord to the underlying document source.

## Implementation status

Implemented and tested today:

- SoftDoc models, MinerU adapter, deterministic pipeline passes, typed Relations, provenance, assets, and validation;
- Exact lookup, SearchUnitBuilder, BM25, multilingual E5 dense retrieval, weighted RRF, SearchSession, and CandidatePreview;
- Planner, Reader, ObservationStore, ExplorationState, EvidenceMemory, Checker-delta, and Answerer contracts and validators.

Still research-stage work:

- the production Controller policy and executable reading loop;
- production-quality Reader and Evidence Checker models;
- deferred planning, Observation Recall, SFT/RL, and full-dataset end-to-end answer evaluation.

The diagram deliberately marks this boundary: it is a system design and contract map, not a claim that the final trained Agent already exists.
