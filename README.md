# AUTO-ML FRAMEWORK

A self-evolving, enterprise-grade ML platform with intelligent agent systems.

## 🎯 MISSION

Build a platform where data scientists are assisted by intelligent agents, and the platform continuously optimizes itself based on usage patterns.

## 🚀 CURRENT STATUS

**Authentication**: ✅ Working (testuser/test123, admin/admin123)
**Current Task**: TASK 9 - 5-AGENT TEAM IMPLEMENTATION

## 📋 FEATURES

### ✅ COMPLETED

- Multi-dataset support with auto-discovery
- Production deployment with monitoring
- Multi-user system with authentication
- Enterprise-grade model registry
- Comprehensive REST API

### 🔄 IN PROGRESS

- DS Agent integration for data scientist assistance
- Architecture Agent foundation for platform optimization
- Continuous platform evolution

## 🏗️ ARCHITECTURE

```
auto-ml/
├── auto_ml/           # Core framework
│   ├── core/         # Abstractions and interfaces
│   ├── data/         # Data ingestion and processing
│   ├── features/     # Feature engineering
│   ├── models/       # Model training and management
│   ├── deployment/   # Production deployment
│   └── monitoring/   # Model monitoring
├── dashboard/        # Web interface
├── projects/         # User project storage
│   └── {user}_{project}_{timestamp}/
│       ├── data/     # Project data
│       ├── models/   # Trained models
│       ├── experiments/ # Experiment tracking
│       ├── results/  # Experiment results
│       ├── config/   # Project configuration
│       ├── deployment/ # Deployment files
│       ├── monitoring/ # Monitoring data
│       └── prompts/  # Project-specific DS agent prompts
│           ├── README.md    # Project overview
│           ├── CONTEXT.md   # Project context
│           ├── TASKS.md     # Project tasks
│           └── GUIDELINES.md # Project guidelines
└── prompts/         # Root-level prompts
    ├── FRAMEWORK_VISION.md      # Main vision and roadmap
    ├── tasks.md                 # Current task tracking
    ├── CORE_AGENT_GUIDE.md      # Core agent (project manager)
    ├── BACKEND_AGENT_GUIDE.md   # Backend agent (server development)
    ├── FRONTEND_AGENT_GUIDE.md  # Frontend agent (UI development)
    ├── TESTING_AGENT_GUIDE.md   # Testing agent (QA and validation)
    ├── DS_AGENT_GUIDE.md        # DS agent (ML and user testing)
    └── README.md                # Documentation overview
```

## 🤖 AGENT TEAM STRUCTURE

### **5-Agent Team with Git Workflow**

#### **CORE AGENT** (master Branch)

- **Role**: Project Manager + Code Reviewer + Integration
- **Focus**: Task design, code review, git merge, coordination
- **Responsibilities**: Coordinate all agents, review PRs, maintain master branch

#### **BACKEND AGENT** (backend Branch)

- **Role**: Backend Development + API + Database
- **Focus**: Server-side logic, APIs, database, core framework
- **Responsibilities**: FastAPI, database, ML pipeline backend

#### **FRONTEND AGENT** (frontend Branch)

- **Role**: UI/UX + Dashboard + User Interface
- **Focus**: Dashboard, user interface, frontend logic
- **Responsibilities**: React/Next.js, user experience, accessibility

#### **TESTING AGENT** (testing Branch)

- **Role**: Quality Assurance + Testing + Validation
- **Focus**: Automated testing, quality assurance, validation
- **Responsibilities**: Test coverage, quality metrics, bug tracking

#### **DS AGENT** (ds-agent Branch)

- **Role**: Data Science + ML + User Testing
- **Focus**: ML pipelines, data science features, initial user testing
- **Responsibilities**: ML development, user testing, project assistance

### **Git Workflow**

```
master (Core Agent)
├── backend (Backend Agent)
├── frontend (Frontend Agent)
├── testing (Testing Agent)
└── ds-agent (DS Agent)
```

**Process**: Core designs tasks → Agents pull and develop → Core reviews and merges

## 🚀 QUICK START

```bash
# Setup
git clone <repository-url>
cd auto-ml
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start API server
python run_api.py

# Start dashboard (new terminal)
cd dashboard
npm install
npm run dev
```

**Demo Access**: testuser/test123, admin/admin123

## 📚 DOCUMENTATION

- **[Framework Vision](prompts/FRAMEWORK_VISION.md)** - Mission, roadmap, and principles
- **[Task Tracking](prompts/tasks.md)** - Current progress and next steps
- **[Core Agent Guide](prompts/CORE_AGENT_GUIDE.md)** - Project management and coordination
- **[Backend Agent Guide](prompts/BACKEND_AGENT_GUIDE.md)** - Server-side development
- **[Frontend Agent Guide](prompts/FRONTEND_AGENT_GUIDE.md)** - UI/UX development
- **[Testing Agent Guide](prompts/TESTING_AGENT_GUIDE.md)** - Quality assurance
- **[DS Agent Guide](prompts/DS_AGENT_GUIDE.md)** - ML development and user testing

## 🔧 DEVELOPMENT

**Core Principles**: Iterative development, real data usage, production readiness, modular design, intelligent evolution

**Task Phases**: Core Framework ✅ → Enterprise Features ✅ → Intelligent Agent System 🔄 → Advanced Intelligence (Future)

---

**Next Milestone**: DS Agent integration for intelligent assistance
