#!/usr/bin/env python3
"""
Prepare industry-specific benchmark datasets for Redukt.

Downloads server logs from Loghub (GitHub) and generates synthetic
e-invoicing XML batches (UBL 2.1, FacturaE 3.2).

Usage: python scripts/prepare-benchmark-data.py
"""

import os
import sys
import urllib.request
import urllib.error

INDUSTRY_DIR = os.path.join("data", "benchmark", "industry")

LOGHUB_BASE = (
    "https://raw.githubusercontent.com/logpai/loghub/master"
)
LOG_FILES = {
    "apache-access.log": f"{LOGHUB_BASE}/Apache/Apache_2k.log",
    "hdfs-audit.log": f"{LOGHUB_BASE}/HDFS/HDFS_v1/HDFS_2k.log",
    "linux-syslog.log": f"{LOGHUB_BASE}/Linux/Linux_2k.log",
    "openssh-auth.log": f"{LOGHUB_BASE}/OpenSSH/OpenSSH_2k.log",
}


def download_logs():
    for name, url in LOG_FILES.items():
        path = os.path.join(INDUSTRY_DIR, name)
        if os.path.exists(path):
            print(f"  [skip] {name} (already exists)")
            continue
        print(f"  [download] {name} ...")
        try:
            urllib.request.urlretrieve(url, path)
            size = os.path.getsize(path)
            print(f"  [ok] {name} ({size:,} bytes)")
        except urllib.error.URLError as e:
            print(f"  [FAIL] {name}: {e}")
            print(f"         URL: {url}")


def generate_ubl_batch():
    """Generate 20 UBL 2.1 invoices (~50 KB batch)."""
    path = os.path.join(INDUSTRY_DIR, "ubl-invoice-batch.xml")
    if os.path.exists(path):
        print(f"  [skip] ubl-invoice-batch.xml (already exists)")
        return

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<InvoiceBatch>"]
    for i in range(1, 21):
        inv = f"""<Invoice xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2"
  xmlns:cac="urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"
  xmlns:cbc="urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2">
  <cbc:UBLVersionID>2.1</cbc:UBLVersionID>
  <cbc:CustomizationID>urn:cen.eu:en16931:2017#compliant#urn:fdc:peppol.eu:2017:poacc:billing:3.0</cbc:CustomizationID>
  <cbc:ProfileID>urn:fdc:peppol.eu:2017:poacc:billing:01:1.0</cbc:ProfileID>
  <cbc:ID>INV-2026-{i:04d}</cbc:ID>
  <cbc:IssueDate>2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}</cbc:IssueDate>
  <cbc:DueDate>2026-{(i % 12) + 1:02d}-{min((i % 28) + 31, 28):02d}</cbc:DueDate>
  <cbc:InvoiceTypeCode>380</cbc:InvoiceTypeCode>
  <cbc:Note>Factura correspondiente al pedido PO-2026-{i * 7:05d}</cbc:Note>
  <cbc:DocumentCurrencyCode>EUR</cbc:DocumentCurrencyCode>
  <cbc:TaxPointDate>2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}</cbc:TaxPointDate>
  <cac:AccountingSupplierParty>
    <cac:Party>
      <cbc:EndpointID schemeID="0190">ES{10000000 + i * 1234:08d}</cbc:EndpointID>
      <cac:PartyIdentification><cbc:ID>SUPP-{i:04d}</cbc:ID></cac:PartyIdentification>
      <cac:PartyName><cbc:Name>Proveedor Ejemplo S.L. Unidad {i}</cbc:Name></cac:PartyName>
      <cac:PostalAddress>
        <cbc:StreetName>Calle Gran Via {i * 3}</cbc:StreetName>
        <cbc:CityName>Madrid</cbc:CityName>
        <cbc:PostalZone>28{i:03d}</cbc:PostalZone>
        <cac:Country><cbc:IdentificationCode>ES</cbc:IdentificationCode></cac:Country>
      </cac:PostalAddress>
      <cac:PartyTaxScheme>
        <cbc:CompanyID>ES{10000000 + i * 1234:08d}</cbc:CompanyID>
        <cac:TaxScheme><cbc:ID>VAT</cbc:ID></cac:TaxScheme>
      </cac:PartyTaxScheme>
    </cac:Party>
  </cac:AccountingSupplierParty>
  <cac:AccountingCustomerParty>
    <cac:Party>
      <cbc:EndpointID schemeID="0190">ES{20000000 + i * 5678:08d}</cbc:EndpointID>
      <cac:PartyName><cbc:Name>Cliente Corporativo {i} S.A.</cbc:Name></cac:PartyName>
      <cac:PostalAddress>
        <cbc:StreetName>Avenida de la Constitucion {i * 5}</cbc:StreetName>
        <cbc:CityName>Barcelona</cbc:CityName>
        <cbc:PostalZone>08{i:03d}</cbc:PostalZone>
        <cac:Country><cbc:IdentificationCode>ES</cbc:IdentificationCode></cac:Country>
      </cac:PostalAddress>
      <cac:PartyTaxScheme>
        <cbc:CompanyID>ES{20000000 + i * 5678:08d}</cbc:CompanyID>
        <cac:TaxScheme><cbc:ID>VAT</cbc:ID></cac:TaxScheme>
      </cac:PartyTaxScheme>
    </cac:Party>
  </cac:AccountingCustomerParty>
  <cac:TaxTotal>
    <cbc:TaxAmount currencyID="EUR">{i * 123.45:.2f}</cbc:TaxAmount>
    <cac:TaxSubtotal>
      <cbc:TaxableAmount currencyID="EUR">{i * 588.33:.2f}</cbc:TaxableAmount>
      <cbc:TaxAmount currencyID="EUR">{i * 123.45:.2f}</cbc:TaxAmount>
      <cac:TaxCategory>
        <cbc:ID>S</cbc:ID>
        <cbc:Percent>21.00</cbc:Percent>
        <cac:TaxScheme><cbc:ID>VAT</cbc:ID></cac:TaxScheme>
      </cac:TaxCategory>
    </cac:TaxSubtotal>
  </cac:TaxTotal>
  <cac:LegalMonetaryTotal>
    <cbc:LineExtensionAmount currencyID="EUR">{i * 588.33:.2f}</cbc:LineExtensionAmount>
    <cbc:TaxExclusiveAmount currencyID="EUR">{i * 588.33:.2f}</cbc:TaxExclusiveAmount>
    <cbc:TaxInclusiveAmount currencyID="EUR">{i * 711.78:.2f}</cbc:TaxInclusiveAmount>
    <cbc:PayableAmount currencyID="EUR">{i * 711.78:.2f}</cbc:PayableAmount>
  </cac:LegalMonetaryTotal>
  <cac:InvoiceLine>
    <cbc:ID>1</cbc:ID>
    <cbc:InvoicedQuantity unitCode="EA">{i * 10}</cbc:InvoicedQuantity>
    <cbc:LineExtensionAmount currencyID="EUR">{i * 294.17:.2f}</cbc:LineExtensionAmount>
    <cac:Item>
      <cbc:Description>Servicio de compresion de datos - Lote {i}</cbc:Description>
      <cbc:Name>Redukt Compression Service</cbc:Name>
    </cac:Item>
    <cac:Price><cbc:PriceAmount currencyID="EUR">{29.417:.3f}</cbc:PriceAmount></cac:Price>
  </cac:InvoiceLine>
  <cac:InvoiceLine>
    <cbc:ID>2</cbc:ID>
    <cbc:InvoicedQuantity unitCode="EA">{i * 5}</cbc:InvoicedQuantity>
    <cbc:LineExtensionAmount currencyID="EUR">{i * 294.16:.2f}</cbc:LineExtensionAmount>
    <cac:Item>
      <cbc:Description>Almacenamiento archival tier - {i * 100} GB</cbc:Description>
      <cbc:Name>Archive Storage</cbc:Name>
    </cac:Item>
    <cac:Price><cbc:PriceAmount currencyID="EUR">{58.832:.3f}</cbc:PriceAmount></cac:Price>
  </cac:InvoiceLine>
</Invoice>"""
        lines.append(inv)
    lines.append("</InvoiceBatch>")
    content = "\n".join(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [ok] ubl-invoice-batch.xml ({len(content):,} bytes, 20 invoices)")


def generate_facturae_batch():
    """Generate 20 FacturaE 3.2.2 invoices (~45 KB batch)."""
    path = os.path.join(INDUSTRY_DIR, "facturae-batch.xml")
    if os.path.exists(path):
        print(f"  [skip] facturae-batch.xml (already exists)")
        return

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<fe:Facturae xmlns:fe="http://www.facturae.gob.es/formato/Versiones/Facturaev3_2_2.xml">',
        "<FileHeader>",
        "  <SchemaVersion>3.2.2</SchemaVersion>",
        "  <Modality>L</Modality>",
        '  <InvoiceIssuerType>EM</InvoiceIssuerType>',
        "  <Batch>",
        "    <BatchIdentifier>ES10000001R-BATCH-2026</BatchIdentifier>",
        "    <InvoicesCount>20</InvoicesCount>",
        "    <TotalInvoicesAmount><TotalAmount>142356.00</TotalAmount></TotalInvoicesAmount>",
        "    <TotalOutstandingAmount><TotalAmount>142356.00</TotalAmount></TotalOutstandingAmount>",
        "    <TotalExecutableAmount><TotalAmount>142356.00</TotalAmount></TotalExecutableAmount>",
        "    <InvoiceCurrencyCode>EUR</InvoiceCurrencyCode>",
        "  </Batch>",
        "</FileHeader>",
        "<Parties>",
        "  <SellerParty>",
        "    <TaxIdentification>",
        "      <PersonTypeCode>J</PersonTypeCode>",
        "      <ResidenceTypeCode>R</ResidenceTypeCode>",
        "      <TaxIdentificationNumber>ES10000001R</TaxIdentificationNumber>",
        "    </TaxIdentification>",
        "    <LegalEntity>",
        "      <CorporateName>Redukt Compression Services S.L.</CorporateName>",
        "      <AddressInSpain>",
        "        <Address>Calle Serrano 41, 2-D</Address>",
        "        <PostCode>28001</PostCode>",
        "        <Town>Madrid</Town>",
        "        <Province>Madrid</Province>",
        "        <CountryCode>ESP</CountryCode>",
        "      </AddressInSpain>",
        "    </LegalEntity>",
        "  </SellerParty>",
        "  <BuyerParty>",
        "    <TaxIdentification>",
        "      <PersonTypeCode>J</PersonTypeCode>",
        "      <ResidenceTypeCode>R</ResidenceTypeCode>",
        "      <TaxIdentificationNumber>ESA28000001</TaxIdentificationNumber>",
        "    </TaxIdentification>",
        "    <LegalEntity>",
        "      <CorporateName>Bankinter S.A.</CorporateName>",
        "      <AddressInSpain>",
        "        <Address>Paseo de la Castellana 29</Address>",
        "        <PostCode>28046</PostCode>",
        "        <Town>Madrid</Town>",
        "        <Province>Madrid</Province>",
        "        <CountryCode>ESP</CountryCode>",
        "      </AddressInSpain>",
        "    </LegalEntity>",
        "  </BuyerParty>",
        "</Parties>",
        "<Invoices>",
    ]

    for i in range(1, 21):
        base = 500.0 + i * 73.42
        tax = base * 0.21
        total = base + tax
        lines.append(f"""  <Invoice>
    <InvoiceHeader>
      <InvoiceNumber>FE-2026-{i:04d}</InvoiceNumber>
      <InvoiceSeriesCode>FE</InvoiceSeriesCode>
      <InvoiceDocumentType>FC</InvoiceDocumentType>
      <InvoiceClass>OO</InvoiceClass>
    </InvoiceHeader>
    <InvoiceIssueData>
      <IssueDate>2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}</IssueDate>
      <InvoiceCurrencyCode>EUR</InvoiceCurrencyCode>
      <TaxCurrencyCode>EUR</TaxCurrencyCode>
      <LanguageName>es</LanguageName>
    </InvoiceIssueData>
    <TaxesOutputs>
      <Tax>
        <TaxTypeCode>01</TaxTypeCode>
        <TaxRate>21.00</TaxRate>
        <TaxableBase><TotalAmount>{base:.2f}</TotalAmount></TaxableBase>
        <TaxAmount><TotalAmount>{tax:.2f}</TotalAmount></TaxAmount>
      </Tax>
    </TaxesOutputs>
    <InvoiceTotals>
      <TotalGrossAmount>{base:.2f}</TotalGrossAmount>
      <TotalGrossAmountBeforeTaxes>{base:.2f}</TotalGrossAmountBeforeTaxes>
      <TotalTaxOutputs>{tax:.2f}</TotalTaxOutputs>
      <TotalTaxesWithheld>0.00</TotalTaxesWithheld>
      <InvoiceTotal>{total:.2f}</InvoiceTotal>
      <TotalOutstandingAmount>{total:.2f}</TotalOutstandingAmount>
      <TotalExecutableAmount>{total:.2f}</TotalExecutableAmount>
    </InvoiceTotals>
    <Items>
      <InvoiceLine>
        <ItemDescription>Servicio mensual de compresion archival - {i * 100} GB procesados</ItemDescription>
        <Quantity>{i * 10}</Quantity>
        <UnitPriceWithoutTax>{base / (i * 10):.6f}</UnitPriceWithoutTax>
        <TotalCost>{base:.2f}</TotalCost>
        <GrossAmount>{base:.2f}</GrossAmount>
        <TaxesOutputs>
          <Tax>
            <TaxTypeCode>01</TaxTypeCode>
            <TaxRate>21.00</TaxRate>
            <TaxableBase><TotalAmount>{base:.2f}</TotalAmount></TaxableBase>
            <TaxAmount><TotalAmount>{tax:.2f}</TotalAmount></TaxAmount>
          </Tax>
        </TaxesOutputs>
      </InvoiceLine>
    </Items>
  </Invoice>""")

    lines.append("</Invoices>")
    lines.append("</fe:Facturae>")
    content = "\n".join(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [ok] facturae-batch.xml ({len(content):,} bytes, 20 invoices)")


def main():
    os.makedirs(INDUSTRY_DIR, exist_ok=True)

    print("=== Redukt Industry Benchmark Data ===\n")

    print("[1/3] Downloading server logs (Loghub)...")
    download_logs()

    print("\n[2/3] Generating UBL 2.1 invoice batch...")
    generate_ubl_batch()

    print("\n[3/3] Generating FacturaE 3.2 invoice batch...")
    generate_facturae_batch()

    print("\n=== Summary ===")
    if os.path.isdir(INDUSTRY_DIR):
        for f in sorted(os.listdir(INDUSTRY_DIR)):
            fp = os.path.join(INDUSTRY_DIR, f)
            if os.path.isfile(fp):
                print(f"  {f:40s} {os.path.getsize(fp):>10,} bytes")

    print("\nDone. Run benchmark-suite.py to benchmark these files.")


if __name__ == "__main__":
    main()
