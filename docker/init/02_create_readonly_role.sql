-- Create read-only role for LLM Text-to-SQL operations
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'llm_readonly') THEN
        CREATE ROLE llm_readonly WITH LOGIN PASSWORD 'readonly_secret';
    END IF;
END
$$;

-- Grant connect on the chinook database
GRANT CONNECT ON DATABASE chinook TO llm_readonly;

-- Grant schema usage
GRANT USAGE ON SCHEMA public TO llm_readonly;

-- Grant SELECT on all existing tables in public schema
GRANT SELECT ON ALL TABLES IN SCHEMA public TO llm_readonly;

-- Column-level Least Privilege: Revoke table-level SELECT on sensitive tables
REVOKE SELECT ON customer, employee FROM llm_readonly;

-- Grant SELECT strictly on non-blocked columns of customer
GRANT SELECT (
    customer_id,
    first_name,
    last_name,
    company,
    city,
    state,
    country,
    support_rep_id
) ON customer TO llm_readonly;

-- Grant SELECT strictly on non-blocked columns of employee
GRANT SELECT (
    employee_id,
    last_name,
    first_name,
    title,
    reports_to,
    hire_date,
    city,
    state,
    country,
    postal_code
) ON employee TO llm_readonly;

-- Grant SELECT on any future tables in public schema
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO llm_readonly;

-- Enforce hard statement timeout of 5 seconds for llm_readonly
ALTER ROLE llm_readonly SET statement_timeout = '5s';

-- Explicitly revoke table and schema creation privileges
REVOKE CREATE ON SCHEMA public FROM llm_readonly;
