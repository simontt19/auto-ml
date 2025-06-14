# Agent Communication & Project Awareness

## Overview

This document defines how agents should interact with the Auto ML framework project and maintain awareness of the current state.

## Project State Awareness

### Primary Documents to Check

1. **`docs/project_state.md`** - Current project status, completed features, next steps
2. **`docs/tasks.md`** - Detailed task tracking and progress logs
3. **`docs/final_goal.md`** - Long-term vision and architecture
4. **`docs/to_user_tasks.md`** - User tasks (only when blockers exist)

### Quick Status Check

When resuming work, always check:

```bash
# Check current git status
git status

# Check project state
cat docs/project_state.md

# Check if user tasks exist
ls docs/to_user_tasks.md
```

## Communication Protocol

### Starting a Session

1. **Read Project State**: Always start by reading `docs/project_state.md`
2. **Check Git Status**: Verify current repository state
3. **Identify Current Phase**: Understand what's in progress
4. **Plan Next Steps**: Based on current state and goals

### During Development

1. **Update State**: Modify `docs/project_state.md` after major changes
2. **Log Progress**: Update `docs/tasks.md` with progress logs
3. **Commit Changes**: Regular commits with descriptive messages
4. **Test Everything**: Run tests after each major change

### When User Help is Needed

1. **Create User Task**: Only when agent cannot proceed
2. **Keep it Simple**: One task at a time, minimal user effort
3. **Delete After Completion**: Remove `docs/to_user_tasks.md` when done
4. **Update State**: Reflect completed user tasks in project state

## Project Structure Awareness

### Key Directories

```
auto_ml/
├── core/           # Core framework (base classes, config, user management)
├── data/           # Data ingestion and processing
├── features/       # Feature engineering
├── models/         # Model training and persistence
├── monitoring/     # Drift detection and monitoring
└── deployment/     # API deployment

docs/               # Documentation and project state
configs/            # Configuration files (credentials, settings)
tests/              # Test suite
projects/           # User project data (when multi-user is active)
```

### Important Files

- **`auto_ml/core/config.py`** - Configuration management
- **`auto_ml/core/user_management.py`** - User authentication
- **`auto_ml/core/pipeline.py`** - Main pipeline orchestration
- **`configs/api_credentials.yaml`** - API tokens and settings
- **`.gitignore`** - Security and file exclusions

## Development Workflow

### 1. Feature Development

```python
# Always follow this pattern:
# 1. Create/update core classes
# 2. Add comprehensive tests
# 3. Update documentation
# 4. Run full test suite
# 5. Update project state
```

### 2. Integration Steps

```python
# When integrating new features:
# 1. Update pipeline orchestration
# 2. Add to API endpoints
# 3. Update configuration
# 4. Test end-to-end
# 5. Update documentation
```

### 3. Testing Protocol

```bash
# Always run tests after changes:
pytest tests/ -v
pytest tests/ --cov=auto_ml --cov-report=html
```

## Configuration Management

### API Credentials

- **Location**: `configs/api_credentials.yaml`
- **Security**: Never commit to git (in .gitignore)
- **Access**: Use `auto_ml.core.config.ConfigManager`

### Environment Settings

- **Development**: Debug mode, detailed logging
- **Production**: Optimized, minimal logging
- **Configuration**: Environment-specific settings in YAML

## User Interaction Protocol

### When User Help is Needed

1. **Identify Blocker**: What specifically requires user action?
2. **Create Simple Task**: One clear, simple task in `docs/to_user_tasks.md`
3. **Explain Why**: Why this requires user intervention
4. **Provide Instructions**: Clear, step-by-step instructions
5. **Wait for Completion**: Don't proceed until user task is done
6. **Clean Up**: Delete user task file after completion

### Example User Task

```markdown
# User Task: Create GitHub Repository

## What's Needed

Create a new GitHub repository for this project.

## Why This is Needed

The agent needs a repository to commit code and track changes.

## Instructions

1. Go to https://github.com
2. Click "New repository"
3. Name it "auto-ml"
4. Make it public
5. Don't initialize with README (agent will handle this)

## Time Required

2 minutes

## Agent Will Handle

- All code development
- Documentation
- Testing
- Configuration
```

## State Management

### Project State Updates

After each major change, update `docs/project_state.md`:

1. **Current Status**: What's currently in progress
2. **Completed Features**: What was just finished
3. **Next Steps**: What comes next
4. **Technical Details**: Architecture changes
5. **User Tasks**: Any pending user actions

### Task Progress Logging

In `docs/tasks.md`, log progress:

```markdown
## Task X Progress Log

- **2024-06-14**: Task started
- **2024-06-14**: Step 1 completed - [description]
- **2024-06-14**: Step 2 completed - [description]
- **2024-06-14**: Task completed successfully
```

## Best Practices

### Code Quality

1. **Type Hints**: Always use type hints
2. **Docstrings**: Comprehensive documentation
3. **Error Handling**: Proper exception handling
4. **Logging**: Appropriate log levels
5. **Testing**: 100% test coverage for new features

### Documentation

1. **Keep Updated**: Update docs with every change
2. **Be Clear**: Write for future agents to understand
3. **Include Examples**: Code examples and usage
4. **Track Progress**: Log all major changes

### Security

1. **Credentials**: Never commit API tokens
2. **Validation**: Validate all inputs
3. **Authentication**: Proper user authentication
4. **Authorization**: Role-based access control

## Communication Checklist

### Before Starting Work

- [ ] Read `docs/project_state.md`
- [ ] Check git status
- [ ] Understand current phase
- [ ] Plan next steps

### During Development

- [ ] Follow coding standards
- [ ] Write comprehensive tests
- [ ] Update documentation
- [ ] Log progress

### After Completing Work

- [ ] Run full test suite
- [ ] Update project state
- [ ] Commit changes
- [ ] Push to repository

### When User Help is Needed

- [ ] Create simple, clear task
- [ ] Explain why it's needed
- [ ] Provide step-by-step instructions
- [ ] Wait for completion
- [ ] Clean up task file

---

**Note**: This document should be updated as the communication protocol evolves. It's essential for maintaining project awareness across different agent sessions.
