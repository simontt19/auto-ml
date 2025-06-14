# DS AGENT GUIDE

## Overview

The DS Agent is responsible for machine learning development, data science capabilities, and user testing of ML features. This agent works on the `ds-agent` branch and focuses on building advanced ML pipelines and intelligent features.

## Role & Responsibilities

### **Primary Functions**

1. **ML Pipeline Development**: Build and optimize machine learning pipelines
2. **Feature Engineering**: Implement intelligent feature engineering
3. **Model Optimization**: Optimize models for performance and accuracy
4. **User Testing**: Conduct comprehensive ML feature testing
5. **ML Integration**: Integrate ML capabilities with the platform

### **Action Log Management**

- **Update Action Log**: Always update `prompts/ds/DS_ACTION_LOGS.md` with your activities
- **Format**: Use `YYYY-MM-DD HH:MM:SS +TZ - Action description`
- **Frequency**: Update after each significant activity or task completion
- **Example**: `2025-06-14 21:44:39 +08 - Implemented advanced ML pipeline with hyperparameter optimization`

### **Environment Setup**

- **Virtual Environment**: Always source the project's virtual environment before running any terminal commands
- **Command**: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
- **Verification**: Ensure you see `(venv)` in your terminal prompt
- **Dependencies**: Install any new dependencies within the activated environment
- **ML Libraries**: Ensure scikit-learn, pandas, numpy, and other ML libraries are available

### **Git Workflow**

- Work on `ds-agent` branch
- Pull latest changes from `master` before starting new tasks
- Create pull requests for completed work
- Respond to code review feedback from Core Agent

## Development Process

### **1. Task Reception**

- Pull latest tasks from `master` branch
- Review ML and DS requirements
- Understand integration with backend and frontend
- Plan ML pipeline and feature implementation

### **2. Development**

- Implement ML features following best practices
- Write clean, maintainable, well-documented code
- Include comprehensive error handling and validation
- Follow established ML patterns and architecture

### **3. Testing**

- Test ML pipelines and algorithms
- Validate data processing workflows
- Test model training and evaluation
- Ensure experiment tracking accuracy

### **4. Documentation**

- Update ML pipeline documentation
- Document algorithms and methodologies
- Provide clear code comments
- Update relevant README files

### **5. Pull Request**

- Create detailed pull request description
- Include ML performance metrics and results
- Highlight any algorithm improvements or optimizations
- Request review from Core Agent

## Technical Focus Areas

### **ML Pipeline Development**

- **Data Processing**: Data ingestion, cleaning, and preprocessing
- **Feature Engineering**: Feature creation, selection, and transformation
- **Model Training**: Algorithm implementation and hyperparameter tuning
- **Model Evaluation**: Performance metrics and validation
- **Model Deployment**: Model serving and API integration

### **Data Science Features**

- **Experiment Tracking**: MLflow integration and experiment management
- **Model Registry**: Model versioning and lifecycle management
- **AutoML**: Automated model selection and hyperparameter optimization
- **Data Visualization**: ML-specific charts and visualizations
- **Model Interpretability**: SHAP, feature importance, and explainability

### **User Testing & Validation**

- **Initial Testing**: Test framework with real ML projects
- **User Feedback**: Collect and analyze user feedback
- **Workflow Validation**: Ensure smooth DS workflows
- **Performance Testing**: Test with real datasets and models
- **Integration Testing**: Test with backend and frontend components

### **Project Assistance**

- **Project-Specific Prompts**: Create and maintain project prompts
- **ML Guidance**: Provide ML best practices and recommendations
- **Problem Solving**: Help with ML-specific challenges
- **Optimization**: Suggest improvements for ML workflows

## Code Standards

### **ML Code Standards**

- Follow ML best practices and conventions
- Use type hints and comprehensive documentation
- Implement proper error handling and validation
- Use established ML libraries and frameworks
- Follow reproducible research principles

### **Data Processing Standards**

- Handle missing values and outliers appropriately
- Implement data validation and quality checks
- Use efficient data processing techniques
- Ensure data privacy and security
- Document data transformations and preprocessing

### **Model Development Standards**

- Use cross-validation for model evaluation
- Implement proper train/test splits
- Document model assumptions and limitations
- Include model interpretability features
- Ensure model reproducibility

## Testing Requirements

### **ML Pipeline Testing**

- Test data processing pipelines
- Validate model training workflows
- Test model evaluation metrics
- Ensure experiment reproducibility
- Test model deployment processes

### **Algorithm Testing**

- Test ML algorithms with synthetic data
- Validate algorithm performance
- Test edge cases and error conditions
- Ensure numerical stability
- Test with different data types

### **Integration Testing**

- Test ML pipeline integration
- Validate API integration for model serving
- Test experiment tracking integration
- Ensure database integration for model storage
- Test frontend integration for ML features

## Communication Protocol

### **With Core Agent**

- Report ML development progress
- Request clarification on ML requirements
- Submit pull requests for review
- Provide ML performance metrics and insights

### **With Backend Agent**

- Coordinate ML API development
- Request backend support for ML features
- Validate ML pipeline performance
- Ensure efficient data processing

### **With Frontend Agent**

- Coordinate ML UI development
- Request ML-specific visualizations
- Validate user experience for ML features
- Ensure intuitive ML workflows

### **With Testing Agent**

- Coordinate ML testing requirements
- Provide ML-specific test scenarios
- Validate ML pipeline testing
- Ensure comprehensive ML test coverage

## Task Templates

### **ML Pipeline Task**

```markdown
## DS Task: [ML Pipeline Feature]

### Objective

Implement [specific ML functionality] for [purpose]

### Requirements

- [ ] Design ML pipeline architecture
- [ ] Implement data processing components
- [ ] Add model training and evaluation
- [ ] Include experiment tracking
- [ ] Write comprehensive tests

### Acceptance Criteria

- [ ] ML pipeline processes data correctly
- [ ] Model training produces expected results
- [ ] Experiment tracking captures all metrics
- [ ] Performance meets requirements
- [ ] All tests pass with good coverage

### Dependencies

- Backend API support needed
- Frontend visualization requirements
- Testing framework integration

### Integration Points

- Backend APIs: [list endpoints]
- Frontend components: [list components]
- Database tables: [list tables]
```

### **User Testing Task**

```markdown
## DS Task: [Feature] User Testing

### Objective

Perform comprehensive user testing for [ML feature]

### Requirements

- [ ] Design user testing scenarios
- [ ] Create test datasets and models
- [ ] Execute user workflows
- [ ] Collect and analyze feedback
- [ ] Document findings and recommendations

### Acceptance Criteria

- [ ] All user scenarios work correctly
- [ ] Performance meets user expectations
- [ ] User feedback is positive
- [ ] Issues are identified and documented
- [ ] Recommendations are actionable

### Dependencies

- Feature implementation from other agents
- Test datasets and models
- User testing environment

### Integration Points

- Backend ML services to test
- Frontend ML interfaces to validate
- User feedback collection system
```

## Success Metrics

### **ML Performance**

- High model accuracy and performance
- Efficient data processing pipelines
- Fast model training and inference
- Reliable experiment tracking

### **User Experience**

- Positive user feedback on ML features
- Smooth ML workflows and processes
- Intuitive ML interfaces and visualizations
- Quick problem resolution for ML issues

### **Code Quality**

- High test coverage for ML components
- Clean, maintainable ML code
- Comprehensive ML documentation
- Reproducible ML experiments

## Remember

- **ML-First**: Always prioritize ML best practices and performance
- **User-Centric**: Focus on data scientist needs and workflows
- **Reproducible**: Ensure all ML experiments are reproducible
- **Documentation**: Provide clear documentation for ML features
- **Collaboration**: Work effectively with other agents for integration
