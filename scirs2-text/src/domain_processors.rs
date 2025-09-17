//! Domain-specific text processors for specialized fields
//!
//! This module provides specialized text processing capabilities for different domains
//! including scientific, legal, and medical text processing with domain-specific
//! vocabularies, entity recognition, and preprocessing pipelines.

use crate::error::{Result, TextError};
use crate::information_extraction::{Entity, EntityType};
use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Domain-specific text processing domains
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Scientific and academic text
    Scientific,
    /// Legal documents and contracts
    Legal,
    /// Medical and clinical text
    Medical,
    /// Financial documents
    Financial,
    /// Patent documents
    Patent,
    /// News and journalism
    News,
    /// Social media content
    SocialMedia,
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Domain::Scientific => write!(f, "scientific"),
            Domain::Legal => write!(f, "legal"),
            Domain::Medical => write!(f, "medical"),
            Domain::Financial => write!(f, "financial"),
            Domain::Patent => write!(f, "patent"),
            Domain::News => write!(f, "news"),
            Domain::SocialMedia => write!(f, "social_media"),
        }
    }
}

/// Configuration for domain-specific processing
#[derive(Debug, Clone)]
pub struct DomainProcessorConfig {
    /// Target domain
    pub domain: Domain,
    /// Whether to preserve technical terms
    pub preserve_technical_terms: bool,
    /// Whether to normalize abbreviations
    pub normalize_abbreviations: bool,
    /// Whether to extract domain-specific entities
    pub extract_entities: bool,
    /// Whether to handle citations and references
    pub handle_citations: bool,
    /// Whether to remove HTML/XML tags
    pub remove_html: bool,
    /// Whether to clean whitespace
    pub clean_whitespace: bool,
    /// Custom stop words for the domain
    pub custom_stop_words: HashSet<String>,
    /// Domain-specific regex patterns
    pub custom_patterns: HashMap<String, String>,
}

impl Default for DomainProcessorConfig {
    fn default() -> Self {
        Self {
            domain: Domain::Scientific,
            preserve_technical_terms: true,
            normalize_abbreviations: true,
            extract_entities: true,
            handle_citations: true,
            remove_html: true,
            clean_whitespace: true,
            custom_stop_words: HashSet::new(),
            custom_patterns: HashMap::new(),
        }
    }
}

/// Scientific text processor
pub struct ScientificTextProcessor {
    config: DomainProcessorConfig,
    citation_regex: Regex,
    formula_regex: Regex,
    chemical_regex: Regex,
    measurement_regex: Regex,
    abbreviation_map: HashMap<String, String>,
    #[allow(dead_code)]
    technical_terms: HashSet<String>,
}

impl ScientificTextProcessor {
    /// Create new scientific text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Scientific citation patterns
        let citation_regex = Regex::new(
            r"\(([A-Za-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?(?:;\s*[A-Za-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?)*)\)"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Mathematical formulas and equations
        let formula_regex =
            Regex::new(r"\$[^$]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]|\\begin\{[^}]+\}.*?\\end\{[^}]+\}")
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Chemical formulas
        let chemical_regex = Regex::new(r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Scientific measurements
        let measurement_regex = Regex::new(
            r"\b\d+(?:\.\d+)?\s*(?:nm|μm|mm|cm|m|km|mg|g|kg|ml|l|°C|°F|K|Pa|kPa|MPa|Hz|kHz|MHz|GHz|V|mV|A|mA|Ω|W|kW|MW)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Common scientific abbreviations
        let mut abbreviation_map = HashMap::new();
        abbreviation_map.insert("e.g.".to_string(), "for example".to_string());
        abbreviation_map.insert("i.e.".to_string(), "that is".to_string());
        abbreviation_map.insert("et al.".to_string(), "and others".to_string());
        abbreviation_map.insert("cf.".to_string(), "compare".to_string());
        abbreviation_map.insert("viz.".to_string(), "namely".to_string());

        // Technical terms to preserve
        let technical_terms = [
            "algorithm",
            "hypothesis",
            "methodology",
            "quantitative",
            "qualitative",
            "statistical",
            "correlation",
            "regression",
            "significance",
            "p-value",
            "standard deviation",
            "confidence interval",
            "meta-analysis",
            "systematic review",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        Ok(Self {
            config,
            citation_regex,
            formula_regex,
            chemical_regex,
            measurement_regex,
            abbreviation_map,
            technical_terms,
        })
    }

    /// Process scientific text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract citations
        if self.config.handle_citations {
            let citation_entities = self.extract_citations_with_positions(&processedtext)?;
            entities.extend(citation_entities);
        }

        // Extract and preserve formulas
        let formulas = self.extract_formulas(&processedtext)?;
        for (i, formula) in formulas.iter().enumerate() {
            let placeholder = format!("[FORMULA_{i}]");
            processedtext = processedtext.replace(formula, &placeholder);
        }
        metadata.insert("formulas".to_string(), formulas.join("|"));

        // Extract measurements
        let measurement_entities = self.extract_measurements_with_positions(&processedtext)?;
        entities.extend(measurement_entities);

        // Extract chemical formulas
        let chemical_entities = self.extract_chemicals_with_positions(&processedtext)?;
        entities.extend(chemical_entities);

        // Normalize abbreviations
        if self.config.normalize_abbreviations {
            for (abbrev, expansion) in &self.abbreviation_map {
                processedtext = processedtext.replace(abbrev, expansion);
            }
        }

        // Clean text while preserving technical terms
        processedtext = self.clean_scientifictext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract citations from text
    #[allow(dead_code)]
    fn extract_citations(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .citation_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract citations with position information
    fn extract_citations_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .citation_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("citation".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract mathematical formulas
    fn extract_formulas(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .formula_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract chemical formulas
    #[allow(dead_code)]
    fn extract_chemicals(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .chemical_regex
            .find_iter(text)
            .filter(|m| {
                let formula = m.as_str();
                // Basic heuristic to filter out non-chemical words
                formula.chars().any(|c| c.is_ascii_uppercase())
                    && formula.chars().any(|c| c.is_ascii_digit())
            })
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract scientific measurements
    #[allow(dead_code)]
    fn extract_measurements(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .measurement_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract measurements with position information
    fn extract_measurements_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .measurement_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("measurement".to_string()),
                confidence: 0.8,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract chemicals with position information
    fn extract_chemicals_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .chemical_regex
            .find_iter(text)
            .filter(|m| {
                let formula = m.as_str();
                // Basic heuristic to filter out non-chemical words
                formula.chars().any(|c| c.is_ascii_uppercase())
                    && formula.chars().any(|c| c.is_ascii_digit())
            })
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("chemical".to_string()),
                confidence: 0.7,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Clean scientific text while preserving important elements
    fn clean_scientifictext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Remove excessive whitespace
        cleaned = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, " ")
            .to_string();

        // Normalize section headers
        cleaned = Regex::new(r"(?i)\b(abstract|introduction|methods?|results?|discussion|conclusion|references?)\s*:?\s*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, |caps: &regex::Captures| {
                format!("[SECTION_{}] ", caps[1].to_uppercase())
            })
            .to_string();

        Ok(cleaned.trim().to_string())
    }
}

/// Legal text processor
pub struct LegalTextProcessor {
    config: DomainProcessorConfig,
    case_citation_regex: Regex,
    statute_regex: Regex,
    #[allow(dead_code)]
    legal_terms: HashSet<String>,
    contract_clauses: Vec<String>,
}

impl LegalTextProcessor {
    /// Create new legal text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Legal case citations
        let case_citation_regex = Regex::new(
            r"\b[A-Z][a-zA-Z\s&,]+v\.?\s+[A-Z][a-zA-Z\s&,]+,?\s*\d+\s+[A-Z][a-z]*\.?\s*\d+(?:\s*\(\d+\))?"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Statute references
        let statute_regex =
            Regex::new(r"\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+|\b\d+\s+C\.?F\.?R\.?\s+§?\s*\d+")
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Legal terminology
        let legal_terms = [
            "plaintiff",
            "defendant",
            "jurisdiction",
            "liability",
            "negligence",
            "contract",
            "tort",
            "damages",
            "injunction",
            "precedent",
            "statute",
            "constitutional",
            "federal",
            "state",
            "court",
            "judge",
            "jury",
            "evidence",
            "testimony",
            "witness",
            "counsel",
            "attorney",
            "lawyer",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        // Common contract clauses
        let contract_clauses = vec![
            "force majeure".to_string(),
            "indemnification".to_string(),
            "limitation of liability".to_string(),
            "intellectual property".to_string(),
            "confidentiality".to_string(),
            "termination".to_string(),
            "governing law".to_string(),
        ];

        Ok(Self {
            config,
            case_citation_regex,
            statute_regex,
            legal_terms,
            contract_clauses,
        })
    }

    /// Process legal text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract case citations
        let case_citation_entities = self.extract_case_citations_with_positions(&processedtext)?;
        entities.extend(case_citation_entities);

        // Extract statute references
        let statute_entities = self.extract_statutes_with_positions(&processedtext)?;
        entities.extend(statute_entities);

        // Identify contract clauses
        let clauses = self.identify_contract_clauses(&processedtext)?;
        metadata.insert("contract_clauses".to_string(), clauses.join("|"));

        // Normalize legal formatting
        processedtext = self.normalize_legaltext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract case citations
    #[allow(dead_code)]
    fn extract_case_citations(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .case_citation_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract case citations with position information
    fn extract_case_citations_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .case_citation_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("case_citation".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract statute references
    #[allow(dead_code)]
    fn extract_statutes(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .statute_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract statute references with position information
    fn extract_statutes_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .statute_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("statute".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Identify contract clauses
    fn identify_contract_clauses(&self, text: &str) -> Result<Vec<String>> {
        let text_lower = text.to_lowercase();
        Ok(self
            .contract_clauses
            .iter()
            .filter(|clause| text_lower.contains(&clause.to_lowercase()))
            .cloned()
            .collect())
    }

    /// Normalize legal text formatting
    fn normalize_legaltext(&self, text: &str) -> Result<String> {
        let mut normalized = text.to_string();

        // Normalize section numbering
        normalized = Regex::new(r"\b(\d+)\.(\d+)\.(\d+)\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&normalized, "Section $1.$2.$3")
            .to_string();

        // Normalize "whereas" clauses
        normalized = Regex::new(r"(?i)\bwhereas\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&normalized, "WHEREAS")
            .to_string();

        Ok(normalized)
    }
}

/// Medical text processor
pub struct MedicalTextProcessor {
    config: DomainProcessorConfig,
    drug_regex: Regex,
    dosage_regex: Regex,
    symptom_regex: Regex,
    #[allow(dead_code)]
    medical_terms: HashSet<String>,
    abbreviations: HashMap<String, String>,
}

impl MedicalTextProcessor {
    /// Create new medical text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Drug name patterns
        let drug_regex =
            Regex::new(r"\b[A-Z][a-z]+(?:mab|nib|tin|pine|pril|sartan|olol|azole|mycin|cillin)\b")
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Dosage patterns
        let dosage_regex =
            Regex::new(r"\b\d+(?:\.\d+)?\s*(?:mg|g|ml|l|units?|tablets?|capsules?|cc)\b")
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Symptom patterns (simplified)
        let symptom_regex = Regex::new(
            r"\b(?:pain|fever|nausea|headache|fatigue|cough|shortness of breath|chest pain)\b",
        )
        .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Medical terminology
        let medical_terms = [
            "diagnosis",
            "prognosis",
            "treatment",
            "therapy",
            "surgery",
            "procedure",
            "symptoms",
            "pathology",
            "etiology",
            "epidemiology",
            "pharmacology",
            "anatomy",
            "physiology",
            "clinical",
            "patient",
            "hospital",
            "physician",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        // Medical abbreviations
        let mut abbreviations = HashMap::new();
        abbreviations.insert("BP".to_string(), "blood pressure".to_string());
        abbreviations.insert("HR".to_string(), "heart rate".to_string());
        abbreviations.insert("RR".to_string(), "respiratory rate".to_string());
        abbreviations.insert("CBC".to_string(), "complete blood count".to_string());
        abbreviations.insert("ECG".to_string(), "electrocardiogram".to_string());
        abbreviations.insert("MRI".to_string(), "magnetic resonance imaging".to_string());
        abbreviations.insert("CT".to_string(), "computed tomography".to_string());

        Ok(Self {
            config,
            drug_regex,
            dosage_regex,
            symptom_regex,
            medical_terms,
            abbreviations,
        })
    }

    /// Process medical text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract drug names
        let drug_entities = self.extract_drugs_with_positions(&processedtext)?;
        entities.extend(drug_entities);

        // Extract dosages
        let dosage_entities = self.extract_dosages_with_positions(&processedtext)?;
        entities.extend(dosage_entities);

        // Extract symptoms
        let symptoms = self.extract_symptoms(&processedtext)?;
        metadata.insert("symptoms".to_string(), symptoms.join("|"));

        // Expand medical abbreviations
        if self.config.normalize_abbreviations {
            for (abbrev, expansion) in &self.abbreviations {
                let pattern = format!(r"\b{}\b", regex::escape(abbrev));
                processedtext = Regex::new(&pattern)
                    .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
                    .replace_all(&processedtext, expansion)
                    .to_string();
            }
        }

        // Clean medical text
        processedtext = self.clean_medicaltext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract drug names
    #[allow(dead_code)]
    fn extract_drugs(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .drug_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract drug names with position information
    fn extract_drugs_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .drug_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("drug".to_string()),
                confidence: 0.8,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract dosage information
    #[allow(dead_code)]
    fn extract_dosages(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .dosage_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract dosage information with position information
    fn extract_dosages_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .dosage_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("dosage".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract symptoms
    fn extract_symptoms(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .symptom_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Clean medical text
    fn clean_medicaltext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Normalize medical record formatting
        cleaned = Regex::new(r"(?i)\b(chief complaint|history of present illness|past medical history|medications|allergies|review of systems|physical examination|assessment|plan)\s*:?\s*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, |caps: &regex::Captures| {
                format!("[{}] ", caps[1].to_uppercase().replace(" ", "_"))
            })
            .to_string();

        Ok(cleaned.trim().to_string())
    }
}

/// Financial text processor
pub struct FinancialTextProcessor {
    config: DomainProcessorConfig,
    currency_regex: Regex,
    financial_instrument_regex: Regex,
    percentage_regex: Regex,
    date_regex: Regex,
    #[allow(dead_code)]
    financial_terms: HashSet<String>,
    #[allow(dead_code)]
    currency_codes: HashSet<String>,
}

impl FinancialTextProcessor {
    /// Create new financial text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Currency patterns
        let currency_regex = Regex::new(
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?|€\d+(?:,\d{3})*(?:\.\d{2})?|£\d+(?:,\d{3})*(?:\.\d{2})?|USD\s*\d+|EUR\s*\d+|GBP\s*\d+"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Financial instruments
        let financial_instrument_regex = Regex::new(
            r"\b(?:bond|stock|share|equity|derivative|option|future|swap|ETF|mutual fund|hedge fund|REIT)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Percentage patterns
        let percentage_regex =
            Regex::new(r"\b\d+(?:\.\d+)?%|percentage|percent|basis points?|bps\b")
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Date patterns (financial context)
        let date_regex = Regex::new(
            r"\b(?:Q[1-4]|quarter)\s*\d{4}|\b\d{1,2}/\d{1,2}/\d{4}|\b\d{4}-\d{2}-\d{2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Financial terminology
        let financial_terms = [
            "revenue",
            "profit",
            "loss",
            "earnings",
            "dividend",
            "yield",
            "volatility",
            "liquidity",
            "capital",
            "assets",
            "liabilities",
            "equity",
            "debt",
            "credit",
            "investment",
            "portfolio",
            "risk",
            "return",
            "valuation",
            "margin",
            "leverage",
            "interest",
            "inflation",
            "gdp",
            "economic",
            "market",
            "trading",
            "broker",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        // Currency codes
        let currency_codes = [
            "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "INR", "BRL",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        Ok(Self {
            config,
            currency_regex,
            financial_instrument_regex,
            percentage_regex,
            date_regex,
            financial_terms,
            currency_codes,
        })
    }

    /// Process financial text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract currency amounts
        let currency_entities = self.extract_currencies_with_positions(&processedtext)?;
        entities.extend(currency_entities);

        // Extract financial instruments
        let instrument_entities =
            self.extract_financial_instruments_with_positions(&processedtext)?;
        entities.extend(instrument_entities);

        // Extract percentages
        let percentages = self.extract_percentages(&processedtext)?;
        metadata.insert("percentages".to_string(), percentages.join("|"));

        // Extract financial dates
        let dates = self.extract_financial_dates(&processedtext)?;
        metadata.insert("financial_dates".to_string(), dates.join("|"));

        // Clean financial text
        processedtext = self.clean_financialtext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract currency amounts
    #[allow(dead_code)]
    fn extract_currencies(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .currency_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract currency amounts with position information
    fn extract_currencies_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .currency_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("currency".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract financial instruments
    #[allow(dead_code)]
    fn extract_financial_instruments(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .financial_instrument_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract financial instruments with position information
    fn extract_financial_instruments_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .financial_instrument_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("financial_instrument".to_string()),
                confidence: 0.8,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract percentages
    fn extract_percentages(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .percentage_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract financial dates
    fn extract_financial_dates(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .date_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Clean financial text
    fn clean_financialtext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Normalize financial section headers
        cleaned = Regex::new(r"(?i)\b(executive summary|financial highlights|income statement|balance sheet|cash flow|notes to financial statements)\s*:?\s*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, |caps: &regex::Captures| {
                format!("[{}] ", caps[1].to_uppercase().replace(" ", "_"))
            })
            .to_string();

        // Normalize currency symbols
        cleaned = cleaned.replace("$", "USD ");
        cleaned = cleaned.replace("€", "EUR ");
        cleaned = cleaned.replace("£", "GBP ");

        Ok(cleaned.trim().to_string())
    }
}

/// Patent text processor for patent documents
pub struct PatentTextProcessor {
    config: DomainProcessorConfig,
}

impl PatentTextProcessor {
    /// Create new patent text processor
    pub fn new(config: DomainProcessorConfig) -> Self {
        Self { config }
    }

    /// Process patent text with domain-specific extraction
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract patent numbers (US patents, EP patents, etc.)
        let patent_number_entities = self.extract_patent_numbers_with_positions(&processedtext)?;
        entities.extend(patent_number_entities);

        // Extract claim numbers and references
        let claim_entities = self.extract_claim_references_with_positions(&processedtext)?;
        entities.extend(claim_entities);

        // Extract technical terms and classifications
        let classification_entities =
            self.extract_patent_classifications_with_positions(&processedtext)?;
        entities.extend(classification_entities);

        // Normalize patent-specific text
        if self.config.normalize_abbreviations {
            processedtext = self.normalize_patenttext(processedtext)?;
        }

        metadata.insert(
            "patent_entities_count".to_string(),
            entities.len().to_string(),
        );
        metadata.insert("processing_domain".to_string(), "patent".to_string());

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: Domain::Patent,
            entities,
            metadata,
        })
    }

    #[allow(dead_code)]
    fn extract_patent_numbers(&self, text: &str) -> Result<Vec<String>> {
        let mut patent_numbers = Vec::new();

        // US patent pattern (e.g., US1234567A1, US1234567B2)
        let us_pattern = Regex::new(r"\bUS\d{7,10}[AB]\d?\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in us_pattern.find_iter(text) {
            patent_numbers.push(mat.as_str().to_string());
        }

        // European patent pattern (e.g., EP1234567A1)
        let ep_pattern = Regex::new(r"\bEP\d{7}[AB]\d?\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in ep_pattern.find_iter(text) {
            patent_numbers.push(mat.as_str().to_string());
        }

        // International application numbers (e.g., WO2021/123456)
        let wo_pattern = Regex::new(r"\bWO\d{4}/\d{6}\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in wo_pattern.find_iter(text) {
            patent_numbers.push(mat.as_str().to_string());
        }

        Ok(patent_numbers)
    }

    fn extract_patent_numbers_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // US patent pattern (e.g., US1234567A1, US1234567B2)
        let us_pattern = Regex::new(r"\bUS\d{7,10}[AB]\d?\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in us_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("patent_number".to_string()),
                confidence: 0.95,
                start: mat.start(),
                end: mat.end(),
            });
        }

        // European patent pattern (e.g., EP1234567A1)
        let ep_pattern = Regex::new(r"\bEP\d{7}[AB]\d?\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in ep_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("patent_number".to_string()),
                confidence: 0.95,
                start: mat.start(),
                end: mat.end(),
            });
        }

        // International application numbers (e.g., WO2021/123456)
        let wo_pattern = Regex::new(r"\bWO\d{4}/\d{6}\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in wo_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("patent_number".to_string()),
                confidence: 0.95,
                start: mat.start(),
                end: mat.end(),
            });
        }

        Ok(entities)
    }

    #[allow(dead_code)]
    fn extract_claim_references(&self, text: &str) -> Result<Vec<String>> {
        let mut claims = Vec::new();

        // Claim references (e.g., "claim 1", "claims 1-5", "dependent claim 3")
        let claim_pattern = Regex::new(r"\b(?:claim|claims)\s+\d+(?:-\d+)?\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in claim_pattern.find_iter(text) {
            claims.push(mat.as_str().to_string());
        }

        Ok(claims)
    }

    fn extract_claim_references_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Claim references (e.g., "claim 1", "claims 1-5", "dependent claim 3")
        let claim_pattern = Regex::new(r"\b(?:claim|claims)\s+\d+(?:-\d+)?\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in claim_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("claim_reference".to_string()),
                confidence: 0.9,
                start: mat.start(),
                end: mat.end(),
            });
        }

        Ok(entities)
    }

    #[allow(dead_code)]
    fn extract_patent_classifications(&self, text: &str) -> Result<Vec<String>> {
        let mut classifications = Vec::new();

        // IPC classifications (e.g., H04L29/06, A61B5/00)
        let ipc_pattern = Regex::new(r"\b[A-H]\d{2}[A-Z]\d{1,3}/\d{2,6}\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in ipc_pattern.find_iter(text) {
            classifications.push(mat.as_str().to_string());
        }

        // CPC classifications
        let cpc_pattern = Regex::new(r"\b[A-H]\d{2}[A-Z]\d{1,3}/\d{2,6}[A-Z]\d*\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in cpc_pattern.find_iter(text) {
            classifications.push(mat.as_str().to_string());
        }

        Ok(classifications)
    }

    fn extract_patent_classifications_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // IPC classifications (e.g., H04L29/06, A61B5/00)
        let ipc_pattern = Regex::new(r"\b[A-H]\d{2}[A-Z]\d{1,3}/\d{2,6}\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in ipc_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("patent_classification".to_string()),
                confidence: 0.85,
                start: mat.start(),
                end: mat.end(),
            });
        }

        // CPC classifications
        let cpc_pattern = Regex::new(r"\b[A-H]\d{2}[A-Z]\d{1,3}/\d{2,6}[A-Z]\d*\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in cpc_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("patent_classification".to_string()),
                confidence: 0.85,
                start: mat.start(),
                end: mat.end(),
            });
        }

        Ok(entities)
    }

    fn normalize_patenttext(&self, text: String) -> Result<String> {
        let mut normalized = text;

        // Normalize common patent abbreviations
        normalized = normalized.replace("fig.", "figure");
        normalized = normalized.replace("Fig.", "Figure");
        normalized = normalized.replace("para.", "paragraph");
        normalized = normalized.replace("ref.", "reference");

        Ok(normalized)
    }
}

/// News text processor for journalism and news content
pub struct NewsTextProcessor {
    config: DomainProcessorConfig,
}

impl NewsTextProcessor {
    /// Create new news text processor
    pub fn new(config: DomainProcessorConfig) -> Self {
        Self { config }
    }

    /// Process news text with domain-specific extraction
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract named entities (people, organizations, locations)
        let person_entities = self.extract_person_names_with_positions(&processedtext)?;
        entities.extend(person_entities);

        // Extract organizations and institutions
        let org_entities = self.extract_organizations_with_positions(&processedtext)?;
        entities.extend(org_entities);

        // Extract dates and time references
        let date_entities = self.extract_dates_with_positions(&processedtext)?;
        entities.extend(date_entities);

        // Extract quotes and attributions
        let quote_entities = self.extract_quotes_with_positions(&processedtext)?;
        entities.extend(quote_entities);

        // Clean and normalize news text
        if self.config.remove_html {
            processedtext = self.clean_news_formatting(processedtext)?;
        }

        metadata.insert(
            "news_entities_count".to_string(),
            entities.len().to_string(),
        );
        metadata.insert("processing_domain".to_string(), "news".to_string());

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: Domain::News,
            entities,
            metadata,
        })
    }

    #[allow(dead_code)]
    fn extract_person_names(&self, text: &str) -> Result<Vec<String>> {
        let mut names = Vec::new();

        // Simple pattern for person names (Title + Name pattern)
        let name_pattern = Regex::new(r"\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.|President|Senator|Rep\.|Gov\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in name_pattern.find_iter(text) {
            names.push(mat.as_str().to_string());
        }

        // Quoted speaker pattern (e.g., "John Smith said")
        let speaker_pattern = Regex::new(
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:said|stated|reported|announced|declared)",
        )
        .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in speaker_pattern.find_iter(text) {
            let speaker = mat
                .as_str()
                .split_whitespace()
                .take_while(|w| w.chars().next().is_some_and(|c| c.is_uppercase()))
                .collect::<Vec<_>>()
                .join(" ");
            if !speaker.is_empty() {
                names.push(speaker);
            }
        }

        Ok(names)
    }

    fn extract_person_names_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Simple pattern for person names (Title + Name pattern)
        let name_pattern = Regex::new(r"\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.|President|Senator|Rep\.|Gov\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in name_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Person,
                confidence: 0.85,
                start: mat.start(),
                end: mat.end(),
            });
        }

        // For speaker patterns, we need to extract just the name part, not the whole match
        let speaker_pattern = Regex::new(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|stated|reported|announced|declared)",
        )
        .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for cap in speaker_pattern.captures_iter(text) {
            if let Some(name_match) = cap.get(1) {
                entities.push(Entity {
                    text: name_match.as_str().to_string(),
                    entity_type: EntityType::Person,
                    confidence: 0.8,
                    start: name_match.start(),
                    end: name_match.end(),
                });
            }
        }

        Ok(entities)
    }

    #[allow(dead_code)]
    fn extract_organizations(&self, text: &str) -> Result<Vec<String>> {
        let mut orgs = Vec::new();

        // Common organization patterns
        let org_suffixes = [
            "Inc\\.",
            "Corp\\.",
            "LLC",
            "Ltd\\.",
            "Co\\.",
            "University",
            "Hospital",
            "Department",
            "Ministry",
            "Agency",
        ];
        for suffix in org_suffixes {
            let pattern = format!(r"\b[A-Z][A-Za-z\s&]+{suffix}");
            let regex = Regex::new(&pattern)
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
            for mat in regex.find_iter(text) {
                orgs.push(mat.as_str().to_string());
            }
        }

        // Government agencies and departments
        let gov_pattern = Regex::new(r"\b(?:FBI|CIA|NASA|FDA|EPA|IRS|Pentagon|White House|Congress|Senate|House of Representatives)\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in gov_pattern.find_iter(text) {
            orgs.push(mat.as_str().to_string());
        }

        Ok(orgs)
    }

    fn extract_organizations_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Common organization patterns
        let org_suffixes = [
            "Inc\\.",
            "Corp\\.",
            "LLC",
            "Ltd\\.",
            "Co\\.",
            "University",
            "Hospital",
            "Department",
            "Ministry",
            "Agency",
        ];
        for suffix in org_suffixes {
            let pattern = format!(r"\b[A-Z][A-Za-z\s&]+{suffix}");
            let regex = Regex::new(&pattern)
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
            for mat in regex.find_iter(text) {
                entities.push(Entity {
                    text: mat.as_str().to_string(),
                    entity_type: EntityType::Organization,
                    confidence: 0.8,
                    start: mat.start(),
                    end: mat.end(),
                });
            }
        }

        // Government agencies and departments
        let gov_pattern = Regex::new(r"\b(?:FBI|CIA|NASA|FDA|EPA|IRS|Pentagon|White House|Congress|Senate|House of Representatives)\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in gov_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Organization,
                confidence: 0.9,
                start: mat.start(),
                end: mat.end(),
            });
        }

        Ok(entities)
    }

    #[allow(dead_code)]
    fn extract_dates(&self, text: &str) -> Result<Vec<String>> {
        let mut dates = Vec::new();

        // Various date formats
        let date_patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{4}\b", // MM/dd/yyyy
            r"\b\d{4}-\d{2}-\d{2}\b",     // yyyy-mm-dd
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            r"\byesterday\b|\btoday\b|\btomorrow\b",
        ];

        for pattern in date_patterns {
            let regex = Regex::new(pattern)
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
            for mat in regex.find_iter(text) {
                dates.push(mat.as_str().to_string());
            }
        }

        Ok(dates)
    }

    fn extract_dates_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Various date formats
        let date_patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{4}\b", // MM/dd/yyyy
            r"\b\d{4}-\d{2}-\d{2}\b",     // yyyy-mm-dd
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            r"\byesterday\b|\btoday\b|\btomorrow\b",
        ];

        for pattern in date_patterns {
            let regex = Regex::new(pattern)
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
            for mat in regex.find_iter(text) {
                entities.push(Entity {
                    text: mat.as_str().to_string(),
                    entity_type: EntityType::Date,
                    confidence: 0.9,
                    start: mat.start(),
                    end: mat.end(),
                });
            }
        }

        Ok(entities)
    }

    #[allow(dead_code)]
    fn extract_quotes(&self, text: &str) -> Result<Vec<String>> {
        let mut quotes = Vec::new();

        // Direct quotes in quotation marks
        let quote_pattern = Regex::new(r#""([^"]{10,200})""#)
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for cap in quote_pattern.captures_iter(text) {
            if let Some(quote) = cap.get(1) {
                quotes.push(quote.as_str().to_string());
            }
        }

        Ok(quotes)
    }

    fn extract_quotes_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Direct quotes in quotation marks
        let quote_pattern = Regex::new(r#""([^"]{10,200})""#)
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for cap in quote_pattern.captures_iter(text) {
            if let Some(quote_match) = cap.get(1) {
                entities.push(Entity {
                    text: quote_match.as_str().to_string(),
                    entity_type: EntityType::Custom("quote".to_string()),
                    confidence: 0.75,
                    start: quote_match.start(),
                    end: quote_match.end(),
                });
            }
        }

        Ok(entities)
    }

    fn clean_news_formatting(&self, text: String) -> Result<String> {
        let mut cleaned = text;

        // Remove bylines and datelines
        let byline_pattern = Regex::new(r"^By\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\n")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        cleaned = byline_pattern.replace_all(&cleaned, "").to_string();

        // Remove wire service tags (AP, Reuters, etc.)
        let wire_pattern = Regex::new(r"\b(?:AP|Reuters|Bloomberg|CNN|BBC)\s*[-–]?\s*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        cleaned = wire_pattern.replace_all(&cleaned, "").to_string();

        Ok(cleaned)
    }
}

/// Social media text processor for social media content
pub struct SocialMediaTextProcessor {
    config: DomainProcessorConfig,
}

impl SocialMediaTextProcessor {
    /// Create new social media text processor
    pub fn new(config: DomainProcessorConfig) -> Self {
        Self { config }
    }

    /// Process social media text with domain-specific extraction
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract hashtags
        let hashtag_entities = self.extract_hashtags_with_positions(&processedtext)?;
        entities.extend(hashtag_entities);

        // Extract mentions (@username)
        let mention_entities = self.extract_mentions_with_positions(&processedtext)?;
        entities.extend(mention_entities);

        // Extract URLs
        let url_entities = self.extract_urls_with_positions(&processedtext)?;
        entities.extend(url_entities);

        // Extract emojis and emoticons
        let emoji_entities = self.extract_emojis_with_positions(&processedtext)?;
        entities.extend(emoji_entities);

        // Clean and normalize social media text
        if self.config.clean_whitespace {
            processedtext = self.normalize_socialtext(processedtext)?;
        }

        metadata.insert(
            "social_entities_count".to_string(),
            entities.len().to_string(),
        );
        metadata.insert("processing_domain".to_string(), "social_media".to_string());

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: Domain::SocialMedia,
            entities,
            metadata,
        })
    }

    #[allow(dead_code)]
    fn extract_hashtags(&self, text: &str) -> Result<Vec<String>> {
        let mut hashtags = Vec::new();

        let hashtag_pattern = Regex::new(r"#\w+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in hashtag_pattern.find_iter(text) {
            hashtags.push(mat.as_str().to_string());
        }

        Ok(hashtags)
    }

    fn extract_hashtags_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        let hashtag_pattern = Regex::new(r"#\w+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in hashtag_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("hashtag".to_string()),
                confidence: 0.95,
                start: mat.start(),
                end: mat.end(),
            });
        }

        Ok(entities)
    }

    #[allow(dead_code)]
    fn extract_mentions(&self, text: &str) -> Result<Vec<String>> {
        let mut mentions = Vec::new();

        let mention_pattern = Regex::new(r"@\w+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in mention_pattern.find_iter(text) {
            mentions.push(mat.as_str().to_string());
        }

        Ok(mentions)
    }

    fn extract_mentions_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        let mention_pattern = Regex::new(r"@\w+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in mention_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("mention".to_string()),
                confidence: 0.9,
                start: mat.start(),
                end: mat.end(),
            });
        }

        Ok(entities)
    }

    #[allow(dead_code)]
    fn extract_urls(&self, text: &str) -> Result<Vec<String>> {
        let mut urls = Vec::new();

        let url_pattern = Regex::new(r"https?://\S+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in url_pattern.find_iter(text) {
            urls.push(mat.as_str().to_string());
        }

        // Also extract shortened URLs
        let short_url_pattern = Regex::new(r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly)/\S+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in short_url_pattern.find_iter(text) {
            urls.push(mat.as_str().to_string());
        }

        Ok(urls)
    }

    fn extract_urls_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        let url_pattern = Regex::new(r"https?://\S+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in url_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("url".to_string()),
                confidence: 0.85,
                start: mat.start(),
                end: mat.end(),
            });
        }

        // Also extract shortened URLs
        let short_url_pattern = Regex::new(r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly)/\S+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        for mat in short_url_pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: EntityType::Custom("url".to_string()),
                confidence: 0.85,
                start: mat.start(),
                end: mat.end(),
            });
        }

        Ok(entities)
    }

    #[allow(dead_code)]
    fn extract_emojis(&self, text: &str) -> Result<Vec<String>> {
        let mut emojis = Vec::new();

        // Simple emoticon patterns
        let emoticon_patterns = [
            r":\)",  // :)
            r":\(",  // :(
            r":D",   // :D
            r":P",   // :P
            r";D",   // ;D
            r"<3",   // <3
            r"</3",  // </3
            r":-\)", // :-)
            r":-\(", // :-(
            r":-D",  // :-D
            r":-P",  // :-P
            r";-\)", // ;-)
        ];

        for pattern in emoticon_patterns {
            let regex = Regex::new(pattern)
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
            for mat in regex.find_iter(text) {
                emojis.push(mat.as_str().to_string());
            }
        }

        // Note: Full Unicode emoji detection would require more complex patterns
        // This is a simplified version for common text-based emoticons

        Ok(emojis)
    }

    fn extract_emojis_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Simple emoticon patterns
        let emoticon_patterns = [
            r":\)",  // :)
            r":\(",  // :(
            r":D",   // :D
            r":P",   // :P
            r";D",   // ;D
            r"<3",   // <3
            r"</3",  // </3
            r":-\)", // :-)
            r":-\(", // :-(
            r":-D",  // :-D
            r":-P",  // :-P
            r";-\)", // ;-)
        ];

        for pattern in emoticon_patterns {
            let regex = Regex::new(pattern)
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
            for mat in regex.find_iter(text) {
                entities.push(Entity {
                    text: mat.as_str().to_string(),
                    entity_type: EntityType::Custom("emoji".to_string()),
                    confidence: 0.8,
                    start: mat.start(),
                    end: mat.end(),
                });
            }
        }

        // Note: Full Unicode emoji detection would require more complex patterns
        // This is a simplified version for common text-based emoticons

        Ok(entities)
    }

    fn normalize_socialtext(&self, text: String) -> Result<String> {
        let mut normalized = text;

        // Convert multiple spaces to single space
        let space_pattern = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;
        normalized = space_pattern.replace_all(&normalized, " ").to_string();

        // Convert common social media abbreviations
        normalized = normalized.replace("u", "you");
        normalized = normalized.replace("ur", "your");
        normalized = normalized.replace("r", "are");
        normalized = normalized.replace("2", "to");
        normalized = normalized.replace("4", "for");

        // Handle repeated characters (e.g., "soooo" -> "so")
        // Using character iteration instead of backreferences
        let chars: Vec<char> = normalized.chars().collect();
        let mut result = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let current_char = chars[i];
            result.push(current_char);

            // Count consecutive occurrences
            let mut count = 1;
            while i + count < chars.len() && chars[i + count] == current_char {
                count += 1;
            }

            // If more than 2 consecutive, add one more (keeping 2 total)
            match count.cmp(&2) {
                std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => {
                    result.push(current_char);
                }
                _ => {}
            }

            i += count;
        }
        normalized = result.into_iter().collect();

        Ok(normalized)
    }
}

/// Result of domain-specific text processing
#[derive(Debug, Clone)]
pub struct ProcessedDomainText {
    /// Original input text
    pub originaltext: String,
    /// Processed text
    pub processedtext: String,
    /// Domain type
    pub domain: Domain,
    /// Extracted domain-specific entities
    pub entities: Vec<Entity>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Unified domain processor that can handle multiple domains
pub struct UnifiedDomainProcessor {
    scientific_processor: Option<ScientificTextProcessor>,
    legal_processor: Option<LegalTextProcessor>,
    medical_processor: Option<MedicalTextProcessor>,
    financial_processor: Option<FinancialTextProcessor>,
    patent_processor: Option<PatentTextProcessor>,
    news_processor: Option<NewsTextProcessor>,
    social_media_processor: Option<SocialMediaTextProcessor>,
}

impl Default for UnifiedDomainProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedDomainProcessor {
    /// Create new unified domain processor
    pub fn new() -> Self {
        Self {
            scientific_processor: None,
            legal_processor: None,
            medical_processor: None,
            financial_processor: None,
            patent_processor: None,
            news_processor: None,
            social_media_processor: None,
        }
    }

    /// Add scientific text processing capability
    pub fn with_scientific(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.scientific_processor = Some(ScientificTextProcessor::new(config)?);
        Ok(self)
    }

    /// Add legal text processing capability
    pub fn with_legal(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.legal_processor = Some(LegalTextProcessor::new(config)?);
        Ok(self)
    }

    /// Add medical text processing capability
    pub fn with_medical(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.medical_processor = Some(MedicalTextProcessor::new(config)?);
        Ok(self)
    }

    /// Add financial text processing capability
    pub fn with_financial(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.financial_processor = Some(FinancialTextProcessor::new(config)?);
        Ok(self)
    }

    /// Add patent text processing capability
    pub fn with_patent(mut self, config: DomainProcessorConfig) -> Self {
        self.patent_processor = Some(PatentTextProcessor::new(config));
        self
    }

    /// Add news text processing capability
    pub fn with_news(mut self, config: DomainProcessorConfig) -> Self {
        self.news_processor = Some(NewsTextProcessor::new(config));
        self
    }

    /// Add social media text processing capability
    pub fn with_social_media(mut self, config: DomainProcessorConfig) -> Self {
        self.social_media_processor = Some(SocialMediaTextProcessor::new(config));
        self
    }

    /// Process text for specified domain
    pub fn process(&self, text: &str, domain: Domain) -> Result<ProcessedDomainText> {
        match domain {
            Domain::Scientific => {
                if let Some(processor) = &self.scientific_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Scientific processor not configured".to_string(),
                    ))
                }
            }
            Domain::Legal => {
                if let Some(processor) = &self.legal_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Legal processor not configured".to_string(),
                    ))
                }
            }
            Domain::Medical => {
                if let Some(processor) = &self.medical_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Medical processor not configured".to_string(),
                    ))
                }
            }
            Domain::Financial => {
                if let Some(processor) = &self.financial_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Financial processor not configured".to_string(),
                    ))
                }
            }
            Domain::Patent => {
                if let Some(processor) = &self.patent_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Patent processor not configured".to_string(),
                    ))
                }
            }
            Domain::News => {
                if let Some(processor) = &self.news_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "News processor not configured".to_string(),
                    ))
                }
            }
            Domain::SocialMedia => {
                if let Some(processor) = &self.social_media_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Social media processor not configured".to_string(),
                    ))
                }
            }
        }
    }

    /// Auto-detect domain from text content
    pub fn detect_domain(&self, text: &str) -> Domain {
        let text_lower = text.to_lowercase();

        // Simple heuristic-based domain detection
        let scientific_keywords = [
            "study",
            "research",
            "hypothesis",
            "methodology",
            "analysis",
            "results",
        ];
        let legal_keywords = [
            "court",
            "law",
            "contract",
            "plaintiff",
            "defendant",
            "statute",
        ];
        let medical_keywords = [
            "patient",
            "diagnosis",
            "treatment",
            "symptoms",
            "medication",
            "clinical",
        ];
        let financial_keywords = [
            "revenue",
            "profit",
            "investment",
            "stock",
            "market",
            "financial",
            "earnings",
            "portfolio",
        ];
        let patent_keywords = [
            "patent",
            "claim",
            "invention",
            "inventor",
            "application",
            "classification",
        ];
        let news_keywords = [
            "reported",
            "said",
            "announced",
            "breaking",
            "journalist",
            "according",
        ];
        let social_keywords = ["#", "@", "retweet", "share", "like", "follow"];

        let sci_score = scientific_keywords
            .iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();

        let legal_score = legal_keywords
            .iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();

        let medical_score = medical_keywords
            .iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();

        let financial_score = financial_keywords
            .iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();

        let patent_score = patent_keywords
            .iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();

        let news_score = news_keywords
            .iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();

        let social_score = social_keywords
            .iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();

        let scores = [
            sci_score,
            legal_score,
            medical_score,
            financial_score,
            patent_score,
            news_score,
            social_score,
        ];
        let max_score = scores.iter().max().unwrap();

        if *max_score == 0 || sci_score == *max_score {
            Domain::Scientific // Default fallback or scientific has max score
        } else if legal_score == *max_score {
            Domain::Legal
        } else if medical_score == *max_score {
            Domain::Medical
        } else if financial_score == *max_score {
            Domain::Financial
        } else if patent_score == *max_score {
            Domain::Patent
        } else if news_score == *max_score {
            Domain::News
        } else {
            Domain::SocialMedia
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scientific_processor() {
        let config = DomainProcessorConfig {
            domain: Domain::Scientific,
            ..Default::default()
        };

        let processor = ScientificTextProcessor::new(config).unwrap();
        let text = "The results show (Smith et al., 2020) that H2O molecules at 25°C demonstrate significant behavior.";

        let result = processor.process(text).unwrap();
        assert_eq!(result.domain, Domain::Scientific);
        assert!(!result.entities.is_empty());
    }

    #[test]
    fn test_medical_processor() {
        let config = DomainProcessorConfig {
            domain: Domain::Medical,
            ..Default::default()
        };

        let processor = MedicalTextProcessor::new(config).unwrap();
        let text = "Patient reports chest pain and was prescribed 10mg aspirin.";

        let result = processor.process(text).unwrap();
        assert_eq!(result.domain, Domain::Medical);
        assert!(!result.entities.is_empty());
    }

    #[test]
    fn test_financial_processor() {
        let config = DomainProcessorConfig {
            domain: Domain::Financial,
            ..Default::default()
        };

        let processor = FinancialTextProcessor::new(config).unwrap();
        let text =
            "The company reported revenue of $1.2 million and stock price increased by 5.3%.";

        let result = processor.process(text).unwrap();
        assert_eq!(result.domain, Domain::Financial);
        assert!(!result.entities.is_empty());
        assert!(result.metadata.contains_key("percentages"));
    }

    #[test]
    fn test_domain_detection() {
        let processor = UnifiedDomainProcessor::new();

        let scientifictext = "This study analyzes the methodology used in the research hypothesis.";
        assert_eq!(processor.detect_domain(scientifictext), Domain::Scientific);

        let legaltext = "The court ruled that the defendant violated the contract law.";
        assert_eq!(processor.detect_domain(legaltext), Domain::Legal);

        let medicaltext = "The patient was diagnosed with symptoms requiring clinical treatment.";
        assert_eq!(processor.detect_domain(medicaltext), Domain::Medical);

        let financialtext = "The portfolio showed strong returns with profit margins increasing and market performance.";
        assert_eq!(processor.detect_domain(financialtext), Domain::Financial);
    }
}
