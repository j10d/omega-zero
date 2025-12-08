# omega-zero
DIY implementation of an AlphaZero-like chess engine for self-education purposes.

This project is being developed with the help of three AI agents:
1) an Architecture Agent
2) an Implementation Agent
3) a Review Agent

The Architecture Agent is simply a chat session where the User interacts with Claude.ai.  The Architecture Agent is the main author of the CLAUDE.md file, which describes the overall project architecture and its components, as well as the component specification files stored in the specs/ folder.  The Implementation and Review Agents are separate Claude Code instances with separate roles and responsibilities defined in the IMPLEMENTATION_AGENT.md and REVIEW_AGENT.md files.  The typical workflow of an implementation and review cycle involves the Implementation and Review Agents completing one of the components prescribed by the Architecture Agent.  At the end of each implementation and review cycle, the Implementation and Review Agents may write notes -- in md files with the prefixes REVIEW_[number] and IMPLEMENTATION_[number] -- documenting their observations and lessons learned.
