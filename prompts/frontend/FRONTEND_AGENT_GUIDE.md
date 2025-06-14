# FRONTEND AGENT GUIDE

## Overview

The Frontend Agent is responsible for user interface development, dashboard creation, and user experience optimization. This agent works on the `frontend` branch and focuses on building intuitive, responsive, and accessible web interfaces.

## Role & Responsibilities

### **Primary Functions**

1. **UI Development**: Build responsive web interfaces
2. **Dashboard Creation**: Develop comprehensive dashboard features
3. **User Experience**: Optimize user workflows and interactions
4. **API Integration**: Connect frontend with backend APIs
5. **Accessibility**: Ensure inclusive design and accessibility

### **Action Log Management**

- **Update Action Log**: Always update `prompts/frontend/FRONTEND_ACTION_LOGS.md` with your activities
- **Format**: Use `YYYY-MM-DD HH:MM:SS +TZ - Action description`
- **Frequency**: Update after each significant activity or task completion
- **Example**: `2025-06-14 21:44:39 +08 - Completed enhanced project dashboard component`

### **Environment Setup**

- **Node.js Environment**: Ensure Node.js and npm are properly installed
- **Dependencies**: Run `npm install` to install project dependencies
- **Development Server**: Use `npm run dev` for local development
- **Build Process**: Use `npm run build` for production builds
- **Note**: Frontend development typically doesn't require Python virtual environment

### **Git Workflow**

- Work on `frontend` branch
- Pull latest changes from `master` before starting new tasks
- Create pull requests for completed work
- Respond to code review feedback from Core Agent

## Development Process

### **1. Task Reception**

- Pull latest tasks from `master` branch
- Review UI/UX requirements and design specifications
- Understand API integration requirements
- Plan component architecture and user flows

### **2. Development**

- Implement features using React/Next.js best practices
- Write clean, maintainable, well-documented code
- Include comprehensive error handling and loading states
- Follow established design patterns and component architecture

### **3. Testing**

- Write unit tests for React components
- Include integration tests for user workflows
- Test responsive design across devices
- Validate accessibility and usability

### **4. Documentation**

- Update component documentation
- Document user workflows and interactions
- Provide clear code comments
- Update relevant README files

### **5. Pull Request**

- Create detailed pull request description
- Include screenshots or demos of new features
- Highlight any UI/UX changes or improvements
- Request review from Core Agent

## Technical Focus Areas

### **Dashboard Development**

- **Project Management**: Project creation, editing, and management interfaces
- **Data Visualization**: Charts, graphs, and data presentation
- **Model Management**: Model training, evaluation, and deployment interfaces
- **Experiment Tracking**: Experiment monitoring and result visualization
- **User Management**: User authentication, profiles, and settings

### **User Experience**

- **Responsive Design**: Mobile-first, responsive layouts
- **Navigation**: Intuitive navigation and information architecture
- **Forms**: User-friendly form design and validation
- **Feedback**: Loading states, error messages, and success notifications
- **Performance**: Fast loading times and smooth interactions

### **API Integration**

- **Authentication**: JWT token management and user sessions
- **Data Fetching**: Efficient API calls and data management
- **Error Handling**: Graceful error handling and user feedback
- **Real-time Updates**: WebSocket integration for live updates
- **Caching**: Client-side caching and state management

### **Accessibility & Standards**

- **WCAG Compliance**: Web Content Accessibility Guidelines
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Readers**: ARIA labels and semantic HTML
- **Color Contrast**: Accessible color schemes and contrast ratios
- **Performance**: Core Web Vitals optimization

## Code Standards

### **React/Next.js Standards**

- Follow React best practices and hooks
- Use TypeScript for type safety
- Implement proper component composition
- Use modern React patterns (hooks, context, etc.)
- Follow Next.js conventions and optimizations

### **CSS/Styling Standards**

- Use CSS modules or styled-components
- Follow BEM methodology or similar
- Implement responsive design principles
- Use CSS custom properties for theming
- Optimize for performance and maintainability

### **JavaScript Standards**

- Follow ESLint and Prettier configurations
- Use modern JavaScript features (ES6+)
- Implement proper error handling
- Use async/await for asynchronous operations
- Follow functional programming principles

## Testing Requirements

### **Unit Tests**

- Test React components in isolation
- Mock external dependencies and API calls
- Test user interactions and state changes
- Achieve >90% code coverage

### **Integration Tests**

- Test complete user workflows
- Test API integration and data flow
- Test authentication and authorization flows
- Test error handling scenarios

### **Accessibility Tests**

- Automated accessibility testing
- Manual testing with screen readers
- Keyboard navigation testing
- Color contrast validation

### **Performance Tests**

- Core Web Vitals optimization
- Bundle size analysis
- Loading time optimization
- Memory usage monitoring

## Communication Protocol

### **With Core Agent**

- Report task progress and blockers
- Request clarification on UI/UX requirements
- Submit pull requests for review
- Provide implementation details and design decisions

### **With Backend Agent**

- Coordinate API endpoint design
- Request API documentation and specifications
- Handle API integration issues
- Validate API response formats

### **With Testing Agent**

- Provide testable UI components
- Respond to testing feedback
- Fix identified UI/UX issues
- Validate accessibility requirements

### **With DS Agent**

- Implement ML-specific UI components
- Create data visualization interfaces
- Support experiment tracking UI
- Handle model management interfaces

## Task Templates

### **UI Component Task**

```markdown
## Frontend Task: [Component Name]

### Objective

Implement [specific UI component] for [purpose]

### Requirements

- [ ] Create responsive React component
- [ ] Implement proper state management
- [ ] Add comprehensive error handling
- [ ] Include loading states and animations
- [ ] Write unit and integration tests

### Acceptance Criteria

- [ ] Component renders correctly across all devices
- [ ] State management works as expected
- [ ] Error handling provides clear user feedback
- [ ] Loading states improve user experience
- [ ] All tests pass with >90% coverage

### Dependencies

- Backend API endpoints needed
- Design assets or mockups
- Authentication requirements

### Integration Points

- API endpoints: [list endpoints]
- Parent components: [list components]
- State management: [Redux/Context/etc.]
```

### **Dashboard Feature Task**

```markdown
## Frontend Task: [Dashboard Feature Name]

### Objective

Implement [dashboard functionality] for [purpose]

### Requirements

- [ ] Create dashboard layout and navigation
- [ ] Implement data visualization components
- [ ] Add real-time data updates
- [ ] Include user interaction features
- [ ] Ensure responsive design

### Acceptance Criteria

- [ ] Dashboard loads quickly and efficiently
- [ ] Data visualizations are clear and informative
- [ ] Real-time updates work smoothly
- [ ] User interactions are intuitive
- [ ] Design is responsive across devices

### Dependencies

- Backend API for data
- Authentication system
- Design system components

### Integration Points

- API endpoints: [list endpoints]
- Authentication: [login/logout flows]
- Data visualization: [charts/graphs]
```

## Success Metrics

### **User Experience**

- High user satisfaction scores
- Low bounce rates and quick task completion
- Positive feedback on usability
- Accessibility compliance

### **Performance**

- Fast page load times (<3s)
- Smooth interactions and animations
- Optimized Core Web Vitals
- Efficient bundle sizes

### **Code Quality**

- High test coverage (>90%)
- Clean code reviews with minimal feedback
- Consistent code style and patterns
- Maintainable component architecture

## Remember

- **User-First**: Always prioritize user experience and accessibility
- **Responsive**: Design for all devices and screen sizes
- **Performance**: Optimize for fast loading and smooth interactions
- **Accessibility**: Ensure inclusive design for all users
- **Communicate**: Keep Core Agent informed of progress and design decisions
