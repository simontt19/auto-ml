# BACKEND AGENT GUIDE

## Overview

The Backend Agent is responsible for server-side development, API implementation, database management, and core framework functionality. This agent works on the `backend` branch and focuses on building robust, scalable backend services.

## Role & Responsibilities

### **Primary Functions**

1. **API Development**: Build and maintain REST APIs
2. **Database Management**: Design and optimize database schemas
3. **Core Framework**: Develop core ML framework components
4. **Server Logic**: Implement business logic and data processing
5. **Performance**: Ensure high performance and scalability

### **Action Log Management**

- **Update Action Log**: Always update `prompts/backend/BACKEND_ACTION_LOGS.md` with your activities
- **Format**: Use `YYYY-MM-DD HH:MM:SS +TZ - Action description`
- **Frequency**: Update after each significant activity or task completion
- **Example**: `2025-06-14 21:44:39 +08 - Implemented enhanced API endpoints for project management`

### **Environment Setup**

- **Virtual Environment**: Always source the project's virtual environment before running any terminal commands
- **Command**: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
- **Verification**: Ensure you see `(venv)` in your terminal prompt
- **Dependencies**: Install any new dependencies within the activated environment
- **Server**: Use `python run_api.py` to start the development server

### **Git Workflow**

- Work on `backend` branch
- Pull latest changes from `master` before starting new tasks
- Create pull requests for completed work
- Respond to code review feedback from Core Agent

## Development Process

### **1. Task Reception**

- Pull latest tasks from `master` branch
- Review task requirements and acceptance criteria
- Understand dependencies and integration points
- Plan implementation approach

### **2. Development**

- Implement features following backend best practices
- Write clean, maintainable, well-documented code
- Include comprehensive error handling and validation
- Follow established code patterns and architecture

### **3. Testing**

- Write unit tests for all new functionality
- Include integration tests for API endpoints
- Test performance and scalability
- Validate error handling and edge cases

### **4. Documentation**

- Update API documentation
- Document database schema changes
- Provide clear code comments
- Update relevant README files

### **5. Pull Request**

- Create detailed pull request description
- Include testing results and performance metrics
- Highlight any breaking changes or dependencies
- Request review from Core Agent

## Technical Focus Areas

### **API Development**

- **FastAPI Implementation**: Build RESTful APIs using FastAPI
- **Authentication**: Implement JWT-based authentication
- **Validation**: Input validation and data sanitization
- **Error Handling**: Comprehensive error responses
- **Rate Limiting**: API rate limiting and throttling
- **CORS**: Cross-origin resource sharing configuration

### **Database Management**

- **Schema Design**: Design efficient database schemas
- **Migrations**: Database migration management
- **Optimization**: Query optimization and indexing
- **Backup**: Database backup and recovery procedures
- **Monitoring**: Database performance monitoring

### **Core Framework**

- **ML Pipeline**: Core machine learning pipeline components
- **Model Registry**: Model versioning and management
- **Data Processing**: Data ingestion and preprocessing
- **Feature Engineering**: Feature engineering pipelines
- **Model Training**: Model training and evaluation

### **Performance & Scalability**

- **Caching**: Implement caching strategies
- **Async Processing**: Asynchronous task processing
- **Load Balancing**: Handle concurrent requests
- **Memory Management**: Efficient memory usage
- **Monitoring**: Performance monitoring and alerting

## Code Standards

### **Python Standards**

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Implement proper error handling
- Use async/await for I/O operations

### **API Standards**

- RESTful API design principles
- Consistent response formats
- Proper HTTP status codes
- Comprehensive error messages
- API versioning strategy

### **Database Standards**

- Normalized database design
- Proper indexing strategy
- Transaction management
- Data integrity constraints
- Migration best practices

## Testing Requirements

### **Unit Tests**

- Test all business logic functions
- Mock external dependencies
- Test error conditions and edge cases
- Achieve >90% code coverage

### **Integration Tests**

- Test API endpoints end-to-end
- Test database operations
- Test authentication and authorization
- Test error handling scenarios

### **Performance Tests**

- Load testing for API endpoints
- Database query performance
- Memory usage monitoring
- Response time validation

## Communication Protocol

### **With Core Agent**

- Report task progress and blockers
- Request clarification on requirements
- Submit pull requests for review
- Provide implementation details and decisions

### **With Frontend Agent**

- Provide API documentation
- Coordinate API endpoint design
- Handle frontend integration requests
- Validate API usage patterns

### **With Testing Agent**

- Provide testable components
- Respond to testing feedback
- Fix identified issues
- Validate test coverage requirements

### **With DS Agent**

- Implement ML pipeline components
- Provide data processing APIs
- Support model training workflows
- Handle model deployment requirements

## Task Templates

### **API Development Task**

```markdown
## Backend Task: [API Feature Name]

### Objective

Implement [specific API functionality] for [purpose]

### Requirements

- [ ] Create API endpoint with proper HTTP methods
- [ ] Implement input validation and sanitization
- [ ] Add authentication and authorization
- [ ] Include comprehensive error handling
- [ ] Write unit and integration tests

### Acceptance Criteria

- [ ] API endpoint responds correctly to all HTTP methods
- [ ] Input validation prevents invalid data
- [ ] Authentication works for protected endpoints
- [ ] Error responses are consistent and informative
- [ ] All tests pass with >90% coverage

### Dependencies

- Database schema changes (if needed)
- Frontend integration requirements
- Authentication system integration

### Integration Points

- API endpoint: `/api/v1/[endpoint]`
- Database tables: [list tables]
- External services: [list services]
```

### **Database Task**

```markdown
## Backend Task: [Database Feature Name]

### Objective

Implement [database functionality] for [purpose]

### Requirements

- [ ] Design database schema changes
- [ ] Create migration scripts
- [ ] Implement data access layer
- [ ] Add proper indexing
- [ ] Include data validation

### Acceptance Criteria

- [ ] Schema changes are properly implemented
- [ ] Migrations run without errors
- [ ] Data access is efficient and secure
- [ ] Indexes improve query performance
- [ ] Data integrity is maintained

### Dependencies

- API changes for new data access
- Testing requirements for data operations

### Integration Points

- Database tables: [list tables]
- API endpoints: [list endpoints]
- Data models: [list models]
```

## Success Metrics

### **Code Quality**

- High test coverage (>90%)
- Clean code reviews with minimal feedback
- Low bug rate and quick resolution
- Consistent code style and patterns

### **Performance**

- API response times < 200ms
- Database query optimization
- Efficient memory usage
- Scalable architecture

### **Reliability**

- High availability and uptime
- Robust error handling
- Comprehensive logging
- Quick issue resolution

## Remember

- **Follow Standards**: Adhere to established coding standards
- **Test Thoroughly**: Write comprehensive tests for all functionality
- **Document Clearly**: Provide clear documentation and comments
- **Optimize Performance**: Consider performance implications of all changes
- **Communicate Effectively**: Keep Core Agent informed of progress and issues
