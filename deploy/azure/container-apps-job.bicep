param location string = resourceGroup().location
param containerAppEnvironmentName string = 'ragtune-env'
param jobName string = 'ragtune-governance-job'
param image string = 'ghcr.io/aim-ragtune/rag-tuning-governance@sha256:PENDING_FIRST_WORKFLOW_RUN'

resource env 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvironmentName
  location: location
}

resource job 'Microsoft.App/jobs@2023-05-01' = {
  name: jobName
  location: location
  properties: {
    environmentId: env.id
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 1800
      replicaRetryLimit: 0
    }
    template: {
      containers: [
        {
          name: 'ragtune'
          image: image
          args: [
            'run-governance-job'
            '--config'
            '/inputs/public_mini_governance_job.yaml'
            '--output-root'
            '/outputs'
            '--decision-out'
            '/outputs/promotion_decision.json'
          ]
          env: [
            {
              name: 'RAGTUNE_STORAGE_MODE'
              value: 'local'
            }
            {
              name: 'RAGTUNE_INPUT_DIR'
              value: '/inputs'
            }
            {
              name: 'RAGTUNE_OUTPUT_DIR'
              value: '/outputs'
            }
          ]
        }
      ]
    }
  }
}
